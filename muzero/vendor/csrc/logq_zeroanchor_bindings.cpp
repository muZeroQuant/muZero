#include <torch/extension.h>

void logq_zeroanchor_quantize_cuda(torch::Tensor grouped, torch::Tensor packed_q, torch::Tensor mn, torch::Tensor scale,
                                   int bits, double alpha, double eps);
void logq_zeroanchor_dequantize_cuda(torch::Tensor packed_q, torch::Tensor mn, torch::Tensor scale, torch::Tensor output,
                                     int bits, double alpha);
void logq_zeroanchor_qk_cuda(torch::Tensor query, torch::Tensor packed_q, torch::Tensor mn, torch::Tensor scale,
                             torch::Tensor output, int bits, double alpha, int num_key_value_groups, double scaling);
void logq_zeroanchor_av_cuda(torch::Tensor attn_weights, torch::Tensor packed_q, torch::Tensor mn, torch::Tensor scale,
                             torch::Tensor output, int bits, double alpha, int num_key_value_groups, int group_size);

void logq_zeroanchor_quantize(torch::Tensor grouped, torch::Tensor packed_q, torch::Tensor mn, torch::Tensor scale,
                              int bits, double alpha, double eps) {
  logq_zeroanchor_quantize_cuda(grouped, packed_q, mn, scale, bits, alpha, eps);
}

void logq_zeroanchor_dequantize(torch::Tensor packed_q, torch::Tensor mn, torch::Tensor scale, torch::Tensor output,
                                int bits, double alpha) {
  logq_zeroanchor_dequantize_cuda(packed_q, mn, scale, output, bits, alpha);
}

void logq_zeroanchor_qk(torch::Tensor query, torch::Tensor packed_q, torch::Tensor mn, torch::Tensor scale,
                        torch::Tensor output, int bits, double alpha, int num_key_value_groups, double scaling) {
  logq_zeroanchor_qk_cuda(query, packed_q, mn, scale, output, bits, alpha, num_key_value_groups, scaling);
}

void logq_zeroanchor_av(torch::Tensor attn_weights, torch::Tensor packed_q, torch::Tensor mn, torch::Tensor scale,
                        torch::Tensor output, int bits, double alpha, int num_key_value_groups, int group_size) {
  logq_zeroanchor_av_cuda(attn_weights, packed_q, mn, scale, output, bits, alpha, num_key_value_groups, group_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quantize", &logq_zeroanchor_quantize, "Zero-anchor LogQ quantize (CUDA)");
  m.def("dequantize", &logq_zeroanchor_dequantize, "Zero-anchor LogQ dequantize (CUDA)");
  m.def("qk", &logq_zeroanchor_qk, "Zero-anchor LogQ fused QK (CUDA)");
  m.def("av", &logq_zeroanchor_av, "Zero-anchor LogQ fused AV (CUDA)");
}