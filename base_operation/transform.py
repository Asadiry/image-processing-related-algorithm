def normalize_to_255(arr):
    arr_min, arr_max = np.min(arr), np.max(arr)
    arr_normalized = (arr - arr_min) / (arr_max - arr_min) * 255
    arr_normalized = arr_normalized.clip(0, 255).astype(np.uint8)
    return arr_normalized