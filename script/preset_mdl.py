def preset_mdl():
    mdl = {
        "is_verbose": 3,  # Verbose level
        # Architectural parameters
        "n_cb": 6,  # Conversion block number
        "n_in_bus": 2,  # Input bus number
        "n_ana_bus": 2,  # Analog bus number
        "n_out_bus": 1,  # Output bus number
        "n_frame": 16,  # Max frame recycling length
        # Nominal value settings
        "K": 1.38e-23,
        "T": 300,
        "v_ref": 0.9,  # Vref
        # SAR weight
        "n_wgt_sar1": [256, 128, 64, 32, 16],
        "n_wgt_sar2": [8, 4, 2, 1, 0.5, 0.25],
        "cu_cdac": 2e-15,
        # Bridge cap
        "cu_bridge": 10e-15,  # Unit cap size of bridge cap
        "n_cu_Cmin1": 5,
        "n_cu_Cmin2": 10,
        "n_cu_Cmaj": 25,
        # Amp gain
        "nominal_amp_gain": 8,  # Nominal amplifier gain
        "nominal_gm_amp": 8e-3,  # Amplifier transconductance
        "nominal_TIA_gain": 1e3,  # TIA gain (current mirror gain x res)
        # Parasitics
        "par_cmp": 10e-15,  # Parasitic capacitance on comparator input
        "par_cbridge_top": 0.05,
        "par_cbridge_bot": 0.1,
        # Non-ideality settings
        "sgm_vn_cmp": 500e-6,  # Input referred rms noise of comparator
        "sgm_vn_amp": 50e-6,  # Input referred rms noise of gm cell (input referred after sampled)
        "sgm_vn_amp_lbw": 25e-6,  # Noise of gm cell under low bandwidth
        "sgm_vos_cmp": 1e-3,  # Input referred offset of comparator
        "sgm_vos_amp": 1e-3,  # Input referred offset of gm cell
        "a_nl2_amp": -60,  # Amp nonlinearity 2nd order coefficient, in dB
        "a_nl3_amp": -60,  # Amp nonlinearity 3rd order coefficient, in dB
        # Non-ideality
        "is_nonideal": 1,
        "is_verbose": 0,
    }

    mdl.update(
        {
            "en_noi_cdac": 1 * mdl["is_nonideal"],  # Noise of CDAC sampling
            "en_noi_cmp": 1 * mdl["is_nonideal"],  # Noise of comparator
            "en_noi_amp": 1 * mdl["is_nonideal"],  # Noise of gm cell
            "en_noi_cfb": 1
            * mdl["is_nonideal"],  # Noise of feedback caps (cmin and cmaj)
            "en_os_cmp": 1 * mdl["is_nonideal"],  # Offset of comparator
            "en_os_amp": 1 * mdl["is_nonideal"],  # Offset of gm cell
            "en_chs_error": 1 * mdl["is_nonideal"],  # Bridge cap parasitics
        }
    )
    return mdl
