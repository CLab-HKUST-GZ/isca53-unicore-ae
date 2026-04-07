
batch_size = 1

cxt_len = [1024, 4096, 8192]
is_generation = False

model_list = [
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Meta-Llama-3-8B",
]

unicore_w4a4 = {
    "pe_dp_size": 1,
    "is_bit_serial": False,
    "pe_energy": 0.123,
    "pe_area": 637289/4096,
    "pe_array_dim": [128, 64],
    
    "w_prec": 4.375,
    "kv_prec": 4,
    "i_prec": 4.25,
}

unicore_w4a8 = {
    "pe_dp_size": 1,
    "is_bit_serial": False,
    "pe_energy": 0.246,
    "pe_area": 637289/2048,
    "pe_array_dim": [128, 32],
    
    "w_prec": 4.375,
    "kv_prec": 4,
    "i_prec": 8,
}

unicore_w8a8 = {
    "pe_dp_size": 1,
    "is_bit_serial": False,
    "pe_energy": 0.246,
    "pe_area": 637289/2048,
    "pe_array_dim": [128, 32],
    
    "w_prec": 8,
    "kv_prec": 8,
    "i_prec": 8,
}

ant_w4a4 = {
    "pe_dp_size": 1,
    "is_bit_serial": False,
    "pe_energy": 0.269,
    "pe_area": 1055044/4096,
    "pe_array_dim": [78, 64],
    
    "w_prec": 4,
    "kv_prec": 16,
    "i_prec": 4,
}

ant_w4a8 = {
    "pe_dp_size": 1,
    "is_bit_serial": False,
    "pe_energy": 0.537,
    "pe_area": 1055044/2048,
    "pe_array_dim": [78, 32],
    
    "w_prec": 4,
    "kv_prec": 16,
    "i_prec": 8,
}

ant_w8a8 = {
    "pe_dp_size": 1,
    "is_bit_serial": False,
    "pe_energy": 1.074,
    "pe_area": 1055044/1024,
    "pe_array_dim": [39, 32],
    
    "w_prec": 8,
    "kv_prec": 16,
    "i_prec": 8,
}

olive_w4a4 = {
    "pe_dp_size": 1,
    "is_bit_serial": False,
    "pe_energy": 0.289,
    "pe_area": 1257427/4096,
    "pe_array_dim": [64, 64],
    
    "w_prec": 4,
    "kv_prec": 16,
    "i_prec": 4,
}

olive_w4a8 = {
    "pe_dp_size": 1,
    "is_bit_serial": False,
    "pe_energy": 0.578,
    "pe_area": 1257427/2048,
    "pe_array_dim": [64, 32],
    
    "w_prec": 4,
    "kv_prec": 16,
    "i_prec": 8,
}

olive_w8a8 = {
    "pe_dp_size": 1,
    "is_bit_serial": False,
    "pe_energy": 1.155,
    "pe_area": 1257427/1024,
    "pe_array_dim": [32, 32],
    
    "w_prec": 8,
    "kv_prec": 16,
    "i_prec": 8,
}

tender_w4a4 = {
    "pe_dp_size": 1,
    "is_bit_serial": False,
    "pe_energy": 0.215,
    "pe_area": 915183/4096,
    "pe_array_dim": [90, 64],
    
    "w_prec": 4,
    "kv_prec": 4,
    "i_prec": 4,
}

tender_w4a8 = {
    "pe_dp_size": 1,
    "is_bit_serial": False,
    "pe_energy": 0.860,
    "pe_area": 915183/1024,
    "pe_array_dim": [45, 32],
    
    "w_prec": 4,
    "kv_prec": 4,
    "i_prec": 8,
}

tender_w8a8 = {
    "pe_dp_size": 1,
    "is_bit_serial": False,
    "pe_energy": 0.860,
    "pe_area": 915183/1024,
    "pe_array_dim": [45, 32],
    
    "w_prec": 8,
    "kv_prec": 8,
    "i_prec": 8,
}

mant_w4a4 = {
    "pe_dp_size": 1,
    "is_bit_serial": False,
    "pe_energy": 0.340,
    "pe_area": 785929/2024,
    "pe_array_dim": [104, 32],
    
    "w_prec": 4.375,
    "kv_prec": 4,
    "i_prec": 4.5,
}

mant_w4a8 = {
    "pe_dp_size": 1,
    "is_bit_serial": False,
    "pe_energy": 0.340,
    "pe_area": 785929/2024,
    "pe_array_dim": [104, 32],
    
    "w_prec": 4.375,
    "kv_prec": 4,
    "i_prec": 8,
}

mant_w8a8 = {
    "pe_dp_size": 1,
    "is_bit_serial": False,
    "pe_energy": 0.681,
    "pe_area": 785929/1024,
    "pe_array_dim": [52, 32],
    
    "w_prec": 8,
    "kv_prec": 8,
    "i_prec": 8,
}


unicore_w16a16 = {
    "pe_dp_size": 1,
    "is_bit_serial": False,
    "pe_energy": 0.123,
    "pe_area": 654.57,
    "pe_array_dim": [128, 16],
    
    "w_prec": 16,
    "kv_prec": 16,
    "i_prec": 16,
}


tender_w16a16 = {
    "pe_dp_size": 1,
    "is_bit_serial": False,
    "pe_energy": 0.123,
    "pe_area": 2659.4,
    "pe_array_dim": [24, 16],
    
    "w_prec": 16,
    "kv_prec": 16,
    "i_prec": 16,
}
