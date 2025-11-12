# Outline for INT3132 Report: Tìm hiểu Flow Matching và Ứng dụng trong Proteomics qua Pep2Prob

## Giới thiệu (Introduction)

- Giới thiệu tổng quát về generative models trong machine learning và ứng dụng trong tin sinh học.
- Nêu vấn đề nghiên cứu: Flow Matching như một framework generative mạnh mẽ, và cách nó có thể hỗ trợ các nhiệm vụ trong proteomics (ví dụ: dự đoán xác suất fragment ion).
- Mục tiêu báo cáo: Cung cấp nền tảng lý thuyết cho thesis, liên kết AI với sinh học.
- Cấu trúc báo cáo.

## Continuous Normalizing Flows (CNFs)

### Generative Models

- Định nghĩa và vai trò của generative models (ví dụ: GANs, VAEs, Flows).
- So sánh với discriminative models.
- Ứng dụng trong tin sinh: Mô hình hóa dữ liệu sinh học phức tạp như protein structures hoặc mass spectra.

### Normalizing Flows

- Định nghĩa: Bijective transformations từ source distribution (e.g., Gaussian) sang target distribution.
- Toán học cơ bản: Push-forward maps, diffeomorphisms (dựa trên Section 3.3 của FM PDF).
- Continuous Normalizing Flows (CNFs): Sử dụng ODE để mô tả flow (Section 3.4).
- Training: Maximum likelihood estimation, simulation-based training (Section 3.7).
- Ưu điểm và hạn chế: Invertible, exact likelihood, nhưng computationally expensive.

## Flow Matching (FM)

- Giới thiệu FM: Framework đơn giản để học velocity fields mà không cần simulation đầy đủ (dựa trên Introduction và Figure 1 của FM PDF).
- Lịch sử phát triển: Từ CNFs đến FM (Lipman et al., 2022; Albergo et al., 2022).

### Key Concepts and Quick Tour

- Probability paths (pt): Interpolation giữa source p và target q (Section 2).
- Velocity fields và ODE simulation (Figure 2).
- So sánh với diffusion models (Section 10).

### Mathematical Foundations

- Random vectors và conditional densities (Section 3.1-3.2).
- Flows as generative models: Continuity Equation (Section 3.5), Instantaneous Change of Variables (Section 3.6).

### Core Flow Matching Algorithm

- Data và building probability paths (Section 4.1-4.2).
- Generating velocity fields (Section 4.3).
- Marginalization Trick (Section 4.4).
- Flow Matching loss (Section 4.5): Regression loss để học velocity.
- Conditional flows và guidance (Section 4.6, 4.10).

### Extensions of Flow Matching

- Optimal Transport và linear/affine conditional flows (Section 4.7-4.8).
- Data couplings (Section 4.9).
- Non-Euclidean FM: Riemannian manifolds (Section 5), ứng dụng trong chemistry/proteins.
- Discrete FM: Continuous Time Markov Chains (CTMC) (Section 6-7).
- General CTMP Models (Section 8).
- Generator Matching: Tổng quát hóa cho multimodal models (Section 9).

### Relation to Other Models

- So sánh với Diffusion Models: Forward/backward process, time-reversal (Section 10).
- Ứng dụng thực tế: Images, videos, audio, proteins (Esser et al., 2024; Huguet et al., 2024).

## Bioinformatics Applications (Bio)

### Proteomics Overview

- Định nghĩa proteomics: Nghiên cứu large-scale về proteins, vai trò trong y học và sinh học (Section 1 của Pep2Prob PDF).
- Tandem Mass Spectrometry (MS2): Quy trình ion hóa, fragmentation, spectra analysis (Figure 1A).
- Thách thức: Peptide identification từ spectra, ảnh hưởng của fragment ion probability (Section 2).
- Current methods: Database search, de novo sequencing, spectral library (Eng et al., 1994; Ma et al., 2003).

### Pep2Prob Benchmark

- Giới thiệu dataset: 608,780 precursors từ 183M spectra (Section 3 của Pep2Prob PDF).
- Problem formulation: Dự đoán P(fragment | precursor) = P(f | p), với f = (ion type, charge, position) (Equations 2.1-2.4, Figure 1B).
- Dataset construction: Annotation, feature representation, ion mask (Section 3.1).
- Train/test split: Ngăn chặn data leakage từ peptides tương tự (Appendix A).
- Baselines: Global statistics vs. peptide-specific models (linear regression, NN, transformer) – Kết quả cho thấy nonlinearities cần ML phức tạp (L1 loss giảm từ 0.24 xuống 0.056).

### Pep2Prob Giúp Bổ Sung cho Flow Matching

- Liên kết: FM có thể mô hình hóa probability paths trong fragmentation process (tương tự CTMP trong FM PDF Section 8-9).
- Ứng dụng tiềm năng: Sử dụng FM để generate fragment probabilities peptide-specific, cải thiện peptide identification/PTM localization (Section 1 và Figure 1 của Pep2Prob).
- Lợi ích: Vượt qua global statistics bằng cách học velocity/generator từ data sinh học (e.g., Riemannian FM cho protein structures).
- Thảo luận: Cách FM có thể được áp dụng vào Pep2Prob (ví dụ: Train velocity field trên spectra paths), và ý nghĩa cho thesis (phần thực nghiệm sau này).

## Kết luận (Conclusion)

- Tóm tắt các khái niệm chính.
- Tiềm năng của Flow Matching trong proteomics: Cải thiện accuracy, mở rộng multimodal models.
- Hướng phát triển cho thesis: Thử nghiệm FM trên Pep2Prob dataset, so sánh với baselines.

## Tài liệu tham khảo (References)

- Liệt kê các paper chính: FM Guide (Lipman et al., 2024), Pep2Prob (Xu et al., 2025), và các reference từ PDFs.

## Phụ lục (Appendix)

- Toán học chi tiết (e.g., Equations từ FM PDF Section 3-4).
- Figures minh họa (e.g., Figure 1 từ cả hai PDFs).
- Code snippets từ flow_matching library (nếu cần, ví dụ: Simple FM implementation).
