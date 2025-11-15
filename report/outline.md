# Outline for INT3132 Report: Tìm hiểu Flow Matching và Ứng dụng trong Proteomics qua Pep2Prob

[Google Docs](https://docs.google.com/document/d/1bWuFuDvJK2Ddm2-ipp9-IHZa9pf3-S9pn3R9O4diMAo/edit?usp=sharing)

## Giới thiệu (Introduction)

- Giới thiệu tổng quát về generative models trong machine learning và ứng dụng trong tin sinh học.
- Nêu vấn đề nghiên cứu: Flow Matching như một framework generative mạnh mẽ, và cách nó có thể hỗ trợ các nhiệm vụ trong proteomics (ví dụ: dự đoán xác suất fragment ion).
- Mục tiêu báo cáo: Cung cấp nền tảng lý thuyết cho thesis, liên kết AI với sinh học.
- Cấu trúc báo cáo.

## Continuous Normalizing Flows (CNFs)

### Generative Models

- Định nghĩa và vai trò của generative models

### Normalizing Flows

- Ý tưởng
- Cơ sở toán học
- Maximum Likelihood
- Residual Flow
- Continuous Normalizing Flows (CNFs): Sử dụng ODE để mô tả flow (Section 3.4).
- Training

## Flow Matching (FM)

### Key Concepts and Quick Tour

- Probability paths (pt): Interpolation giữa source p và target q
- Velocity fields và ODE simulation.

### Mathematical Foundations

- Random vectors và conditional densities
- Flows as generative models: Continuity Equation (Section 3.5), Instantaneous Change of Variables (Section 3.6).

### Core Flow Matching Algorithm

- Data và building probability paths
- Generating velocity fields
- Marginalization Trick
- Flow Matching loss
- Conditional flows và guidance

## Bioinformatics Applications (Bio)

### Proteomics Overview

### Pep2Prob Benchmark

### Pep2Prob Giúp Bổ Sung cho Flow Matching

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
