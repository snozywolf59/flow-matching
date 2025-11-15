## Generative Model

Giả sử chúng ta có các mẫu $x_1, x2, …, x_n$ từ 1 phân phối $q(x)$, trong đó $q(x)$ là phân phối mà ta chưa biết. Từ các mẫu xi này, ta sẽ muốn tạo ra một mô hình học xác suất xấp xỉ với $q(x)$. Đây được gọi là generative model, với ý tưởng cốt lõi là cố gắng học quá trình sinh dữ liệu ngoài thực tế.
<br>
Generative model là một lĩnh vực quan trọng trong trí tuệ nhân tạo và học máy, tập trung vào việc xây dựng các mô hình có khả năng tạo ra dữ liệu mới dựa trên phân phối của dữ liệu huấn luyện. Các kỹ thuật phổ biến bao gồm Generative Adversarial Networks (GANs), nơi một mạng generator tạo dữ liệu giả và một mạng discriminator phân biệt thật-giả để cải thiện chất lượng; Variational Autoencoders (VAEs) sử dụng không gian ẩn để tái tạo dữ liệu; và gần đây là diffusion models, dần dần thêm nhiễu rồi loại bỏ để sinh mẫu mới. Những mô hình này giúp vượt qua hạn chế của dữ liệu hạn chế, mở ra tiềm năng sáng tạo vô hạn.
<br>
Ứng dụng của generative modeling ngày càng rộng rãi, từ lĩnh vực hình ảnh như tạo ảnh thực tế (ví dụ: Stable Diffusion) đến văn bản (như GPT models cho viết lách tự động), âm nhạc (tạo giai điệu mới), và y tế (mô phỏng hình ảnh MRI để hỗ trợ chẩn đoán). Trong khoa học dữ liệu, chúng hỗ trợ tăng cường dữ liệu (data augmentation) để cải thiện mô hình học máy. Tuy nhiên, thách thức lớn là kiểm soát chất lượng đầu ra và tránh lạm dụng, như tạo deepfakes. Generative modeling không chỉ thúc đẩy sáng tạo mà còn định hình tương lai của AI, với tiềm năng ứng dụng trong thiết kế sản phẩm, giải trí và nghiên cứu khoa học.

## Normalizing Flows

Normalizing Flows (NF) là một mô hình sinh xác suất (generative model) sử dụng một chuỗi các phép biến đổi khả nghịch (invertible transformations) để ánh xạ một phân phối xác suất dễ lây mẫu (gọi là phân phối cơ sở, $q_0$) thành một phân phối xác suất phức tạp hơn nhằm xấp xỉ phân phối kỳ vọng, là phân phối thực tế của dữ liệu ($p_1$).

### Ý tưởng (Idea)

Ý tưởng cốt lõi của Normalizing Flows là sử dụng một hàm biến đổi khả vi liên tục và khả nghịch $\phi: \mathbb{R}^d \rightarrow \mathbb{R}^d$, được tham số hóa bằng mạng nơ-ron ($\phi_\theta$), để chuyển đổi mẫu từ một phân phối ban đầu đơn giản ($q_0$, ví dụ: phân phối Gaussian chuẩn $\mathcal{N}(0, I)$) thành các mẫu thuộc phân phối phức tạp của dữ liệu ($p_1$).
<br>
Mục tiêu là tối ưu hóa các tham số $\theta$ của hàm biến đổi $\phi_\theta$ sao cho phân phối $p_1$ được tạo ra phân phối kỳ vọng sát nhất với phân phối dữ liệu thực tế.

### Cơ sở toán học

Cơ sở toán học của Normalizing Flows là **Change-of-Variable Formula** cho mật độ xác suất.

Nếu $x \sim q_0$ và $y = \phi(x)$, mật độ xác suất của $y$, ký hiệu là $p_1(y)$, được tính như sau:

$$p_1(y) = q_0(\phi^{-1}(y)) \left|\det\left[\frac{\partial \phi^{-1}}{\partial y}(y)\right]\right|$$

Trong đó:

1.  $\phi(x)$ là hàm biến đổi phân phối cơ sở thành phân phối kỳ vọng. Hàm này khả vi liên tục và khả nghịch.
2.  $\frac{\partial \phi^{-1}}{\partial y}(y)$ là **Ma trận Jacobian** của ánh xạ ngược $\phi^{-1}$.
3.  $\left|\det\left[\cdot\right]\right|$ là giá trị tuyệt đối của định thức Jacobian.

Để dễ tính toán hơn, công thức thường được viết lại dưới dạng:

$$p_1(y) = \frac{q_0(x)}{\left|\det\left[\frac{\partial \phi}{\partial x}(x)\right]\right|} \quad \text{với } x = \phi^{-1}(y)$$

### learning Params by maximum likelyhood

Phương pháp tiêu chuẩn để đào tạo các tham số $\theta$ của một Normalizing Flow $\phi_\theta$ là sử dụng **Ước lượng Khả năng Hợp lý Cực đại (Maximum Likelihood Estimation - MLE)**.

- **Mục tiêu:** Tối đa hóa xác suất mà mô hình gán cho dữ liệu huấn luyện $\mathcal{D}$.
  $$\textrm{argmax}_{\theta}\ \ \mathbb{E}_{x\sim \mathcal{D}} [\log p_1(x)]$$
  trong đó $p_1(x)$ là mật độ xác suất do flow $\phi_\theta$ tạo ra.
- **Tính toán:** $\log p_1(x)$ được tính bằng công thức đổi biến (đã đề cập ở phần trước):
  $$\log p_1(y) = \log q_0(\phi^{-1}(y)) - \log \left|\det\left[\frac{\partial \phi}{\partial x}(x)\right]\right|$$
- **Thách thức:** Để tối ưu hóa hàm mục tiêu này, mô hình phải giải quyết ba vấn đề kỹ thuật lớn:
  1.  Đảm bảo hàm biến đổi $\phi_\theta$ khả nghịch (Invertible).
  2.  Có thể tính toán hàm ngược $\phi^{-1}$ một cách hiệu quả.
  3.  Có thể tính toán định thức Jacobian $\det[\partial \phi/\partial x]$.

### Residual Flows

Residual Flow là một lớp các hàm biến đổi $\phi_k$ được thiết kế để giải quyết vấn đề tính toán Jacobian một cách hiệu quả ...

Một Residual Flow có dạng:
$$\phi_k(x) = x + \delta \ u_k(x)$$
Trong đó:

- $x$ là đầu vào.
- $u_k(x)$ là một hàm dư (residual connection), thường được tham số hóa bằng mạng nơ-ron.
- $\delta$ là một hằng số nhỏ dương.

Flow nhận được:
$$\phi = \phi_K \circ \ldots \circ \phi_2 \circ \phi_1.$$

Flow trên có likelihôd được tính bởi tổng likelihood của các flow thành phần:
$$\log q(y) = \log p(\phi^{-1}(y)) + \sum_{k=1}^K \log \det\left[\frac{\partial \phi_k^{-1}}{\partial x_{k+1}}(x_{k+1})\right]$$

### Continuous Normalizing Flows (CNFs): Sử dụng ODE để mô tả flow

Như đã nói ở trên, Residual Flow là chuỗi các phép biến đổi $\phi(x) = x + \delta \ u(x)$ với $\delta > 0$. Suy ra:
$$\frac{\phi(x) - x}{\delta} = u(x)$$

Continuous Normalizing Flows (CNFs) là hàm giới hạn khi cho số lượng residual flow tiến tới vô hạn trong 1 khoảng thời gian. Trong CNFs, phép biến đổi $\phi$ được mô tả bằng Phương trình Vi phân (ODE).

#### Mô tả Flow bằng ODE

Nếu một Residual Flow có dạng $\phi(x) = x + \delta u(x)$, khi $\delta \to 0$ (hay số lớp biến đổi $K \to \infty$), phép biến đổi được mô tả bởi một trường vectơ $u_t(x_t)$:

$$\frac{d x_t}{d t} = u_t(x_t)$$

Trong đó:

- $t \in [0, 1]$ là tham số thời gian.
- $x_t$ là điểm dữ liệu tại thời điểm $t$.
- $u_t(x_t)$ là trường vectơ do mạng nơ-ron $u_\theta$ tham số hóa, mô tả tốc độ thay đổi của $x_t$.
- $x_1$ (tại $t=1$) là mẫu dữ liệu cuối cùng, và $x_0$ (tại $t=0$) là mẫu từ phân phối cơ sở.

#### Công thức log-density trong CNFs

Sự thay đổi mật độ xác suất $\log p_t(x_t)$ theo thời gian $t$ được tính thông qua công thức **Liouville** (một dạng của Phương trình Vận chuyển) bằng cách sử dụng **độ phân kỳ (divergence)** của trường vectơ $u_t$:

$$\frac{\dd}{\dd t} \log p_t(x_t) = - (\nabla \cdot u_t)(x_t) = - \mathrm{div}\ u_t(x_t)$$

Từ đó, log-density của phân phối dữ liệu $p_1(x)$ (tại $t=1$) được tính bằng cách tích phân độ phân kỳ theo thời gian:

$$\log p_\theta(x) = \log p_0(x_0) - \int_0^1 (\nabla \cdot u_\theta)(x_t) \dd t$$

Quá trình này chỉ yêu cầu tính toán độ phân kỳ (divergence) của trường vectơ $u_\theta$, thay vì toàn bộ định thức Jacobian của một phép biến đổi phức tạp, giúp việc tính toán hiệu quả hơn.

### Training

#### Mục tiêu tối ưu

Việc huấn luyện Normalizing Flows thường dựa trên nguyên tắc Ước lượng Khả năng Hợp lý Cực đại (Maximum Likelihood Estimation - MLE). Mục tiêu là tìm tham số $\theta$ của flow $\phi_\theta$ để tối đa hóa log-khả năng hợp lý của dữ liệu $\mathcal{D}$ dưới mô hình:

$$\textrm{argmax}_{\theta}\ \ \mathbb{E}_{x\sim \mathcal{D}} [\log p_1(x)]$$

Trong đó, $\log p_1(x)$ được tính bằng Công thức Đổi biến (đối với flows rời rạc) hoặc công thức tích phân độ phân kỳ (đối với CNFs).

#### 

- Flows Rời rạc
- CNFs