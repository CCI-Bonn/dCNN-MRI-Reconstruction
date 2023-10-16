# Function for the forward operator A in the paper
def piA(x, mask, nrow, ncol):
    """
    This function represents the forward operator A, as defined in the paper.
    It takes an image x and a binary mask and returns the k-space data.
    """
    ccImg = np.reshape(x, (nrow, ncol))
    kspace = np.fft.fft2(ccImg) / np.sqrt(nrow * ncol)
    res = kspace[mask != 0]
    return res

# Function for the adjoint operator A^T in the paper
def piAt(kspaceUnder, mask, nrow, ncol):
    """
    This function represents the adjoint operator A^T, as defined in the paper.
    It takes undersampled k-space data, a binary mask, and image dimensions
    and returns the reconstructed image.
    """
    temp = np.zeros((nrow, ncol), dtype=np.complex64)
    if len(mask.shape) == 2:
        mask = np.tile(mask, (1, 1))

    temp[mask != 0] = kspaceUnder
    img = np.fft.ifft2(temp) * np.sqrt(nrow * ncol)
    return img.astype(np.complex64)

# Function to generate undersampled k-space data
def generateUndersampled(org, mask, sigma=0.):
    """
    This function generates undersampled k-space data from an original image.
    It takes the original image, a binary mask, and an optional noise level
    and returns the undersampled k-space data.
    """
    nrow, ncol, nSlice = org.shape
    atb = np.empty(org.shape, dtype=np.complex64)
    for i in range(nSlice):
        A = lambda z: piA(z, mask[:, :, i], nrow, ncol)
        At = lambda z: piAt(z, mask[:, :, i], nrow, ncol)
        sidx = np.where(mask[:, :, i].ravel() != 0)[0]
        nSIDX = len(sidx)
        noise = np.random.randn(nSIDX,) + 1j * np.random.randn(nSIDX,)
        noise = noise * (sigma / np.sqrt(2.))
        y = A(org[:, :, i]) + noise
        atb[:, :, i] = At(y)
    return atb

# Function to crop a subvolume from the input data
def cropdata(inp, patch_size):
    """
    This function crops a subvolume from the input data.
    It takes the input data and the desired patch size and returns the cropped subvolume.
    """
    nrow, ncol, nSlice = inp.shape
    x = np.random.randint(nrow - patch_size[0] + 1)
    y = np.random.randint(ncol - patch_size[1] + 1)
    z = np.random.randint(nSlice - patch_size[2] + 1)
    return inp[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]]
