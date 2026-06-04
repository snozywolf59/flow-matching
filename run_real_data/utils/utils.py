import matplotlib.pyplot as plt
from time import time
import numpy as np
import torch
import torch.nn.functional as F
from pyteomics import mass
from spectrum_utils import spectrum
from spectrum_utils.plot import spectrum as plot_spectrum
from spectrum_utils.plot import mirror as plot_mirror

from typing import Union

PROSIT_ALHABET = {
    "A": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "K": 9,
    "L": 10,
    "M": 11,
    "N": 12,
    "P": 13,
    "Q": 14,
    "R": 15,
    "S": 16,
    "T": 17,
    "V": 18,
    "W": 19,
    "Y": 20,
    "M(ox)": 21,
}
PROSIT_INDEXED_ALPHABET = {i: c for c, i in PROSIT_ALHABET.items()}


def get_peptide_seq(integer_seq):
    return "".join(
        PROSIT_INDEXED_ALPHABET[int(i)] for i in integer_seq if i != 0
    )


def plot_loss_history(
    loss_history,
    val_loss_history=None,
    title="Loss History",
    prefix="Loss_History",
    smooth_window=None,
):
    """
    Plot training/validation loss history.

    Args:
        loss_history (list): train loss history
        val_loss_history (list, optional): validation loss history
        smooth_window (int, optional): moving average window
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from time import time

    plt.figure(figsize=(8, 5))

    # Train loss
    plt.plot(loss_history, label="Train Loss")

    if smooth_window is not None and smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        smooth_loss = np.convolve(
            np.array(loss_history),
            kernel,
            mode="valid",
        )

        plt.plot(
            range(smooth_window - 1, len(loss_history)),
            smooth_loss,
            label=f"Train Smooth ({smooth_window})",
        )

    # Validation loss
    if val_loss_history is not None:
        plt.plot(
            val_loss_history,
            label="Validation Loss",
        )

    plt.xlabel("Log Step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(f"{prefix}_{time()}.jpg", bbox_inches="tight")
    plt.show()


def to_numpy(x):
    try:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def plot_intensity(
    intensity_vector: Union[torch.Tensor, np.ndarray],
    peptide,
    precursor_charge: int,
    allow_invalid: bool = True,
):
    """
    Plot ProteomeTools-style intensity with structural masking.

    Args:
        intensity_vector (torch.Tensor | np.ndarray): shape (174,)
        peptide (list | np.ndarray): integer peptide sequence
        precursor_charge (int): precursor charge
        allow_invalid (bool): whether to visualize invalid fragments
    """

    if not isinstance(intensity_vector, (torch.Tensor, np.ndarray)):
        raise TypeError("intensity_vector must be torch.Tensor or np.ndarray")

    # convert safely
    if isinstance(intensity_vector, torch.Tensor):
        x = intensity_vector.detach().cpu().numpy().copy()
    else:
        x = intensity_vector.copy()

    if x.ndim != 1:
        raise ValueError("intensity_vector must be 1D")

    # create structural mask
    mask = create_fragment_mask_from_peptide(peptide, precursor_charge)

    if len(x) != len(mask):
        raise ValueError("Intensity vector must have length 174")

    # mark structural invalid as -1
    x[mask == 0] = -1

    indices = np.arange(len(x))
    valid_mask = x != -1
    invalid_mask = x == -1

    plt.figure(figsize=(12, 4))

    if allow_invalid:
        plt.scatter(indices[valid_mask], x[valid_mask], s=15, label="Valid")
        plt.scatter(
            indices[invalid_mask],
            x[invalid_mask],
            marker="x",
            s=20,
            label="Invalid",
        )

        plt.ylim(-1.1, 1.05)
    else:
        plt.scatter(indices[valid_mask], x[valid_mask], s=15)
        plt.ylim(0, 1.05)

    plt.xlabel("Fragment Index")
    plt.ylabel("Normalized Intensity")
    plt.title("Fragment Intensity Vector (Masked)")
    plt.grid(alpha=0.3)
    if allow_invalid:
        plt.legend()
    plt.tight_layout()
    plt.show()


def create_fragment_mask_from_peptide(
    peptide, precursor_charge, max_len=30, max_frag_charge=3
):
    """
    Create mask for ProteomeTools FI intensity vector (174 dims).

    Args:
        peptide (list or np.ndarray): integer-encoded peptide sequence
        precursor_charge (int): precursor charge
        max_len (int): maximum peptide length (default=30)
        max_frag_charge (int): maximum fragment charge (default=3)

    Returns:
        np.ndarray: mask vector of shape (174,)
    """

    seq_len = np.count_nonzero(peptide)
    assert seq_len <= max_len
    assert precursor_charge >= 1

    n_pos = max_len - 1
    mask = np.zeros(2 * max_frag_charge * n_pos)

    valid_positions = seq_len - 1

    idx = 0
    for pos in range(n_pos):

        pos_valid = pos < valid_positions

        for ion_type in range(2):  # b,y
            for frag_charge in range(1, max_frag_charge + 1):

                if pos_valid and frag_charge <= precursor_charge:
                    mask[idx] = 1

                idx += 1

    return mask


def create_batch_fragment_mask_from_peptide(
    batch_peptide,
    batch_precursor_charge,
    max_len=30,
    max_frag_charge=3,
    reshape=True,
):
    """
    Create mask for ProteomeTools FI intensity vector ((max_len-1) * num_frag_charge * 2 dims).

    Args:
        peptide (list or np.ndarray): (B, L) integer-encoded peptide sequence
        precursor_charge (int): (B,) or (B, 1) precursor charge
        max_len (int): maximum peptide length (default=30)
        max_frag_charge (int): maximum fragment charge (default=3)

    Returns:
        np.ndarray: mask vector of shape (B, maxlen, 2*max_frag_charge) or (B, maxlen-1, 2*max_frag_charge)
    """

    batch_size = batch_peptide.shape[0]
    n_pos = max_len - 1
    mask = np.zeros((batch_size, 2 * max_frag_charge * n_pos))

    for i in range(batch_size):
        seq_len = np.count_nonzero(batch_peptide[i])
        valid_positions = seq_len - 1

        idx = 0
        for pos in range(n_pos):

            pos_valid = pos < valid_positions

            for ion_type in range(2):  # b,y
                for frag_charge in range(1, max_frag_charge + 1):

                    if pos_valid and frag_charge <= batch_precursor_charge[i]:
                        mask[i, idx] = 1

                    idx += 1
    if reshape:
        return mask.reshape(batch_size, max_len - 1, 2 * max_frag_charge)
    return mask


def process_intensity_vector(
    intensity_vector: Union[torch.Tensor, np.ndarray], reshape
) -> Union[torch.Tensor, np.ndarray]:
    # turn (B, 174)  into (B, 29, 6) with 6 channels: y1, y2, y3, b1, b2, b3
    batch_size = intensity_vector.shape[0]
    if intensity_vector.shape[-1] != 174:
        raise ValueError("Input vector must have length 174")

    # intensity_vector[intensity_vector < 0] = 0

    if isinstance(intensity_vector, torch.Tensor):
        if reshape:
            return intensity_vector.reshape(batch_size, 29, 6).clamp()
        return intensity_vector

    elif isinstance(intensity_vector, np.ndarray):
        if reshape:
            return intensity_vector.reshape(batch_size, 29, 6)
        return intensity_vector
    else:
        raise TypeError("Input must be torch.Tensor or np.ndarray")


def masked_mse_loss(pred, target, mask=None, eps=1e-8):
    assert pred.shape == target.shape, "pred and target must have same shape"
    loss = (pred - target) ** 2

    if mask is None:
        return loss.mean()
    assert (
        mask.shape == pred.shape
    ), "mask must have same shape with pred and target"

    mask = mask.float()
    return (loss * mask).sum() / (mask.sum() + eps)


def logit_transform(x, alpha=0.05):
    x = alpha + (1 - 2 * alpha) * x
    return torch.log(x) - torch.log(1 - x)


def unlogit_transform(y, alpha=0.05):
    x = torch.sigmoid(y)
    x = (x - alpha) / (1 - 2 * alpha)
    return x


def calculate_mzs(
    peptide_sequence, charge=2, max_len: int = 30, max_frag_charge: int = 3
):
    # print(f"Peptide: {peptide_sequence} | Max Charge: {charge}")
    # print(f"{'Ion':<10} | {'m/z':<10}")
    # print("-" * 25)
    L = len(peptide_sequence)
    # Tính toán các loại ion b và y
    # b_series: từ đầu N-terminus
    # y_series: từ đầu C-
    mzs = np.zeros(2 * (max_len - 1) * max_frag_charge)
    for i in range(1, L):
        for z in range(1, min(max_frag_charge + 1, charge + 1)):
            # Tính ion b
            b_mz = mass.fast_mass(peptide_sequence[:i], ion_type="b", charge=z)
            # print(f"b{i}^{z}+    | {b_mz:.4f}")
            mzs[(i - 1) * 6 + max_frag_charge + z - 1] = b_mz
            # Tính ion y
            y_mz = mass.fast_mass(peptide_sequence[i:], ion_type="y", charge=z)
            # print(f"y{L - i}^{z}+    | {y_mz:.4f}")
            mzs[(L - i - 1) * 6 + z - 1] = y_mz
    return mzs


def plot_intensity_spectrum(seq, charge, intensities):
    precursor_mz = mass.fast_mass(seq, charge=charge)
    mzs = calculate_mzs(seq, charge)
    spec = spectrum.MsmsSpectrum(
        identifier=f"{seq}_{charge}",
        precursor_mz=precursor_mz,
        precursor_charge=charge,
        mz=mzs,
        intensity=intensities,
    )
    plot_spectrum(spec)
    plt.show()


def plot_intensity_mirror(
    seq,
    charge,
    intensities_top,
    intensities_bottom,
    label_top="Predicted",
    label_bottom="Ground Truth",
):
    precursor_mz = mass.fast_mass(seq, charge=charge)
    mzs = calculate_mzs(seq, charge)

    spec_top = spectrum.MsmsSpectrum(
        identifier=f"{seq}_{charge}_top",
        precursor_mz=precursor_mz,
        precursor_charge=charge,
        mz=mzs,
        intensity=intensities_top,
    )

    spec_bot = spectrum.MsmsSpectrum(
        identifier=f"{seq}_{charge}_bot",
        precursor_mz=precursor_mz,
        precursor_charge=charge,
        mz=mzs,
        intensity=intensities_bottom,
    )

    plot_mirror(spec_top, spec_bot)
    plt.show()
