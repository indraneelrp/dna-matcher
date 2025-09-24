#include "kseq/kseq.h"
#include "common.h"

struct MatchResultDevice {
	char sample_name[64];
	char signature_name[64];
	double match_score;
	int integrity_hash;
};

__device__ void deviceStrncpy(char* dest, const char* src, size_t n) {
	for (size_t i = 0; i < n; i++) {
		dest[i] = src[i];
		if (src[i] == '\0') break;
	}
	if (n > 0) dest[n-1] = '\0';
}

__global__ void matcherKernel(char **sample_names, int *sample_name_lens, char **sample_seqs, int *sample_seq_lens, char **sample_quals, char **sig_names, int *sig_name_lens, char **sig_seqs, int *sig_seq_lens, MatchResultDevice *results, int *match_count, int numSamples, int numSigs) {
	int sampleIdx = blockIdx.x;
	int sigIdx = threadIdx.x;

	if (sampleIdx >= numSamples || sigIdx >= numSigs) return;

	char *sample_name = sample_names[sampleIdx];
	// int sample_name_len = sample_name_lens[sampleIdx];
	char *sample_seq = sample_seqs[sampleIdx];
	int sample_seq_len = sample_seq_lens[sampleIdx];
	char *sample_qual = sample_quals[sampleIdx];

	char *sig_name = sig_names[sigIdx];
	// int sig_name_len = sig_name_lens[sigIdx];
	char *sig_seq = sig_seqs[sigIdx];
	int sig_seq_len = sig_seq_lens[sigIdx];

	bool match = false;
	double best_match_score = 0;
	for (int i = 0; i < sample_seq_len - sig_seq_len; i++) {
		int j;
		for (j = 0; j < sig_seq_len; j++) {
			if (sample_seq[i+j] != 'N' && sig_seq[j] != 'N' && sample_seq[i+j] != sig_seq[j]) break;
		}
		if (j == sig_seq_len) {
			match = true;
			double curr_match_score = 0;
			for (int k = i; k < i+j; k++) {
				double k_phred_score = 0;
				if (sample_seq[k] != 'N') {
					k_phred_score = double(sample_qual[k]) - 33;
				}
				curr_match_score += k_phred_score;
			}
			curr_match_score /= sig_seq_len;
			if (curr_match_score >= best_match_score) {
				best_match_score = curr_match_score;
			}
		}
	}
	
	if (match == true) {
		int sample_integrity_hash = 0;
		for (int i = 0; i < sample_seq_len; i++) {		
			double i_phred_score = 0;
			if (sample_seq[i] != 'N') {
				i_phred_score = double(sample_qual[i]) - 33;
			}
			sample_integrity_hash += i_phred_score;
		}
		sample_integrity_hash %= 97;

		int result_idx = atomicAdd(match_count, 1);
		deviceStrncpy(results[result_idx].sample_name, sample_name, 64);
		deviceStrncpy(results[result_idx].signature_name, sig_name, 64);
		results[result_idx].match_score = best_match_score;
		results[result_idx].integrity_hash = sample_integrity_hash;
	}
}

void runMatcher(const std::vector<klibpp::KSeq>& samples, const std::vector<klibpp::KSeq>& signatures, std::vector<MatchResult>& matches) {
	// this is the "main" fn that we run the device fn from
	// set number of blocks per grid and number of threads per block
	int numSamples = samples.size();	// blocks per grid
	int numSigs = signatures.size();
	int blocksPerGrid = 2200;
	//int threadsPerBlock = 1024;
	int threadsPerBlock = ((numSigs+31)/32) * 32;

	// allocate arrays for pointers
	char **h_sample_names = new char*[numSamples];
	int *h_sample_name_lens = new int[numSamples];
	char **h_sample_seqs = new char*[numSamples];
	int *h_sample_seq_lens = new int[numSamples];
	char **h_sample_quals = new char*[numSamples];

	char **h_sig_names = new char*[numSigs];
	int *h_sig_name_lens = new int[numSigs];
	char **h_sig_seqs = new char*[numSigs];
	int *h_sig_seq_lens = new int[numSigs];

	// allocate memory for GPU (samples, sigs: name, name_len, seq, seq_len, qual. flattened result array). 
	char **d_sample_names, **d_sample_seqs, **d_sample_quals, **d_sig_names, **d_sig_seqs;
	int *d_sample_name_lens, *d_sample_seq_lens, *d_sig_name_lens, *d_sig_seq_lens, *d_match_count;
	MatchResultDevice *d_results;

	cudaMalloc(&d_sample_names, numSamples * sizeof(char *));
	cudaMalloc(&d_sample_seqs, numSamples * sizeof(char *));
	cudaMalloc(&d_sample_quals, numSamples * sizeof(char *));
	cudaMalloc(&d_sample_name_lens, numSamples * sizeof(int));
	cudaMalloc(&d_sample_seq_lens, numSamples * sizeof(int));

	cudaMalloc(&d_sig_names, numSigs * sizeof(char *));
	cudaMalloc(&d_sig_seqs, numSigs * sizeof(char *));
	cudaMalloc(&d_sig_name_lens, numSigs * sizeof(int));
	cudaMalloc(&d_sig_seq_lens, numSigs * sizeof(int));

	cudaMalloc(&d_results, numSamples * numSigs * sizeof(MatchResultDevice));
	cudaMalloc(&d_match_count, sizeof(int));


	// copy samples and signatures to GPU
	for (int i = 0; i < numSamples; i++) {
		h_sample_name_lens[i] = samples[i].name.size();
		h_sample_seq_lens[i] = samples[i].seq.size();

		cudaMalloc(&h_sample_names[i], h_sample_name_lens[i] * sizeof(char));
		cudaMemcpy(h_sample_names[i], samples[i].name.c_str(), h_sample_name_lens[i] * sizeof(char), cudaMemcpyHostToDevice);

		cudaMalloc(&h_sample_seqs[i], h_sample_seq_lens[i] * sizeof(char));
		cudaMemcpy(h_sample_seqs[i], samples[i].seq.c_str(), h_sample_seq_lens[i] * sizeof(char), cudaMemcpyHostToDevice);

		cudaMalloc(&h_sample_quals[i], h_sample_seq_lens[i] * sizeof(char));
		cudaMemcpy(h_sample_quals[i], samples[i].qual.c_str(), h_sample_seq_lens[i] * sizeof(char), cudaMemcpyHostToDevice);
	}

	for (int i = 0; i < numSigs; i++) {
		h_sig_name_lens[i] = signatures[i].name.size();
		h_sig_seq_lens[i] = signatures[i].seq.size();

		cudaMalloc(&h_sig_names[i], h_sig_name_lens[i] * sizeof(char));
		cudaMemcpy(h_sig_names[i], signatures[i].name.c_str(), h_sig_name_lens[i] * sizeof(char), cudaMemcpyHostToDevice);

		cudaMalloc(&h_sig_seqs[i], h_sig_seq_lens[i] * sizeof(char));
		cudaMemcpy(h_sig_seqs[i], signatures[i].seq.c_str(), h_sig_seq_lens[i] * sizeof(char), cudaMemcpyHostToDevice);
	}


	// copy pointers to GPU
	cudaMemcpy(d_sample_names, h_sample_names, numSamples * sizeof(char *), cudaMemcpyHostToDevice);
	cudaMemcpy(d_sample_name_lens, h_sample_name_lens, numSamples * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_sample_seqs, h_sample_seqs, numSamples * sizeof(char *), cudaMemcpyHostToDevice);
	cudaMemcpy(d_sample_seq_lens, h_sample_seq_lens, numSamples * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_sample_quals, h_sample_quals, numSamples * sizeof(char *), cudaMemcpyHostToDevice);


	cudaMemcpy(d_sig_names, h_sig_names, numSigs * sizeof(char *), cudaMemcpyHostToDevice);
	cudaMemcpy(d_sig_name_lens, h_sig_name_lens, numSigs * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_sig_seqs, h_sig_seqs, numSigs * sizeof(char *), cudaMemcpyHostToDevice);
	cudaMemcpy(d_sig_seq_lens, h_sig_seq_lens, numSigs * sizeof(int), cudaMemcpyHostToDevice);
	
	int zero = 0;
	cudaMemcpy(d_match_count, &zero, sizeof(int), cudaMemcpyHostToDevice);


	// run kernel function then cudeDeviceSynchronize()
	matcherKernel<<<blocksPerGrid, threadsPerBlock>>>(d_sample_names, d_sample_name_lens, d_sample_seqs, d_sample_seq_lens, d_sample_quals, d_sig_names, d_sig_name_lens, d_sig_seqs, d_sig_seq_lens, d_results, d_match_count, numSamples, numSigs);
	cudaDeviceSynchronize();

	int h_match_count;
	cudaMemcpy(&h_match_count, d_match_count, sizeof(int), cudaMemcpyDeviceToHost);


	// copy results from kernel function into matches
	MatchResultDevice *h_results = new MatchResultDevice[h_match_count];
	cudaMemcpy(h_results, d_results, h_match_count * sizeof(MatchResultDevice), cudaMemcpyDeviceToHost);


	// process results
	matches.clear();
	for (int i = 0; i < h_match_count; i++) {
		matches.push_back({
				std::string(h_results[i].sample_name),
				std::string(h_results[i].signature_name),
				h_results[i].match_score,
				h_results[i].integrity_hash });
	}


	// free memory
	for (int i = 0; i < numSamples; i++) {
		cudaFree(h_sample_names[i]);
		cudaFree(h_sample_seqs[i]);
		cudaFree(h_sample_quals[i]);
	}
	for (int j = 0; j < numSigs; j++) {
		cudaFree(h_sig_names[j]);
		cudaFree(h_sig_seqs[j]);
	}
	delete[] h_sample_names;
	delete[] h_sample_name_lens;
	delete[] h_sample_seqs;
	delete[] h_sample_seq_lens;
	delete[] h_sample_quals;
	delete[] h_sig_names;
	delete[] h_sig_name_lens;
	delete[] h_sig_seqs;
	delete[] h_sig_seq_lens;;
	delete[] h_results;

	cudaFree(d_sample_names);
	cudaFree(d_sample_name_lens);
	cudaFree(d_sample_seqs);
	cudaFree(d_sample_seq_lens);
	cudaFree(d_sample_quals);
	cudaFree(d_sig_names);
	cudaFree(d_sig_name_lens);
	cudaFree(d_sig_seqs);
	cudaFree(d_sig_seq_lens);
	cudaFree(d_results);
	cudaFree(d_match_count);
}
