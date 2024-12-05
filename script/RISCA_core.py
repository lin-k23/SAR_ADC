import numpy as np
import re


def RISCA_core(mdl, pr, sig):
    # -------------------------- setup ----------------------------------------
    v_in_p = sig[0]
    v_in_n = sig[1]
    task_ch = pr["task_ch"]
    task_conf = pr["T_assembler"][:task_ch]

    KT = mdl["K"] * mdl["T"]

    n_bit_sar1 = len(mdl["n_wgt_sar1"])
    n_bit_sar2 = len(mdl["n_wgt_sar2"])

    n_wgt_sum = np.sum(mdl["n_wgt_sar1"]) + np.sum(mdl["n_wgt_sar2"])
    a_wgt_sar1 = mdl["n_wgt_sar1"] / n_wgt_sum
    a_wgt_sar2 = mdl["n_wgt_sar2"] / n_wgt_sum

    phy_cap_CDAC = n_wgt_sum * mdl["cu_cdac"]  # total CDAC capacitance
    phy_cap_Cmaj = (
        mdl["n_cu_Cmaj"]
        * mdl["cu_bridge"]
        * (1 + mdl["par_cbridge_top"] * mdl["en_chs_error"])
        + mdl["par_cmp"] * mdl["en_chs_error"]
    )
    phy_cap_Cmin1 = (
        mdl["n_cu_Cmin1"]
        * mdl["cu_bridge"]
        * (1 + mdl["par_cbridge_top"] * mdl["en_chs_error"])
    )  # C-min1
    phy_cap_Cmin2 = (
        mdl["n_cu_Cmin2"]
        * mdl["cu_bridge"]
        * (1 + mdl["par_cbridge_top"] * mdl["en_chs_error"])
    )  # C-min2

    # non-ideality settings
    sgm_vn_cmp = (
        mdl["sgm_vn_cmp"] * np.ones(task_ch) * mdl["en_noi_cmp"]
    )  # input referred rms noise of comparator
    sgm_vn_amp = (
        mdl["sgm_vn_amp"] * np.ones(task_ch) * mdl["en_noi_amp"]
    )  # input referred rms noise of gm cell (input referred after sampled)
    sgm_vn_amp_lbw = (
        mdl["sgm_vn_amp_lbw"] * np.ones(task_ch) * mdl["en_noi_amp"]
    )  # noise of gm cell under low bandwidth

    a_nl2_amp = (
        10 ** (mdl["a_nl2_amp"] / 20)
        * 8
        * 2
        / 0.05625
        * np.ones(task_ch)
        * mdl["is_nonideal"]
    )  # gm cell nonlinearity (2nd order coefficient)
    a_nl3_amp = (
        10 ** (mdl["a_nl3_amp"] / 20)
        * 8
        * 4
        / (0.05625**2)
        * np.ones(task_ch)
        * mdl["is_nonideal"]
    )  # gm cell nonlinearity (3rd order coefficient)

    # -------------------------- initialize ----------------------------------
    v_cdac_p = np.zeros(task_ch)
    v_cdac_n = np.zeros(task_ch)
    v_cmaj_p = np.zeros(task_ch)
    v_cmaj_n = np.zeros(task_ch)
    v_cmin1_p = np.zeros(task_ch)
    v_cmin1_n = np.zeros(task_ch)
    v_cmin2_p = np.zeros(task_ch)
    v_cmin2_n = np.zeros(task_ch)
    v_cmp = np.zeros(task_ch)

    v_cmp_os = (
        np.random.randn(task_ch) * mdl["sgm_vos_cmp"] * mdl["en_os_cmp"]
    )  # input referred offset of comparator
    v_amp_os = (
        np.random.randn(task_ch) * mdl["sgm_vos_amp"] * mdl["en_os_amp"]
    )  # input referred offset of gm cell
    # v_cmp_os = np.ones(task_ch) * 50e-3  # input referred offset of comparator
    # v_amp_os = np.ones(task_ch) * 50e-3  # input referred offset of gm cell

    i_self_p = np.zeros(task_ch)
    i_self_n = np.zeros(task_ch)

    i_bus_p = np.zeros(mdl["n_ana_bus"])
    i_bus_n = np.zeros(mdl["n_ana_bus"])

    d_out_ch = np.zeros((task_ch, n_bit_sar1 + n_bit_sar2))
    rdym_ch = np.zeros(task_ch)
    rdyl_ch = np.zeros(task_ch)

    # ------------------------- run simulation -------------------------------
    iter_out = 0
    iter_frame = 0

    _, n_sim = v_in_p.shape
    d_out_list = []

    for iter_sim in range(1, n_sim + 1):
        if mdl["is_verbose"] >= 1:
            print(f"\niter_sim={iter_sim}")

        iter_frame = (iter_frame % pr["task_frame"]) + 1

        N_TASK = 8  # looping of N_TASK is for priority
        for iter_task in range(1, N_TASK + 1):
            for iter_ch in range(1, pr["task_ch"] + 1):
                conf_col = task_conf.iloc[iter_ch - 1, iter_frame - 1]

                if conf_col is None:
                    continue

                if iter_task == 1:  # output DOUT, reset SAR or bridge cap -------------
                    # output data
                    if "OUT" in conf_col:
                        iter_out += 1

                        # 16-bit digital output:
                        # 3 addr, 2 rdy, 11 digital output bits
                        addr = [0, 0, 0]
                        addr[0] = (iter_ch - 1) // 4
                        addr[1] = (iter_ch - addr[0] * 4) // 2
                        addr[2] = iter_ch - addr[0] * 4 - addr[1] * 2

                        d_out_raw = [0] * 16
                        d_out_raw[0:3] = addr
                        d_out_raw[3:5] = [rdym_ch[iter_ch - 1], rdyl_ch[iter_ch - 1]]
                        if rdyl_ch[iter_ch - 1] == 1:
                            d_out_raw[5:16] = d_out_ch[iter_ch - 1, :]
                        else:
                            d_out_raw[11:16] = d_out_ch[iter_ch - 1, 0:5]
                        d_out_list.append(d_out_raw)

                        if mdl["is_verbose"] >= 2:
                            print(
                                f"[OUT][{iter_ch-1}] output digital code = \n{d_out[iter_out][5:16]}"
                            )
                            print(f"[OUT][{iter_ch-1}] d_out_ch = \n{d_out_ch}")

                    # reset SAR
                    if "RST" in conf_col:
                        d_out_ch[iter_ch - 1, :] = 0
                        rdym_ch[iter_ch - 1] = 0
                        rdyl_ch[iter_ch - 1] = 0

                    # reset Cmaj or Cmin
                    if "RMAJ" in conf_col:
                        v_cmaj_p[iter_ch - 1] = 0
                        v_cmaj_n[iter_ch - 1] = 0
                    if "RMIN2" in conf_col:
                        v_cmin2_p[iter_ch - 1] = 0
                        v_cmin2_n[iter_ch - 1] = 0

                    # reset bridge cap
                    reg_pattern = r"SRT\d+"
                    match = re.search(reg_pattern, conf_col)
                    if match:
                        pos_start, pos_end = match.span()
                        Ctot = 0
                        if "0" in conf_col[pos_start + 3 : pos_end]:
                            Ctot += phy_cap_Cmaj
                        if "1" in conf_col[pos_start + 3 : pos_end]:
                            Ctot += phy_cap_Cmin1
                        if "2" in conf_col[pos_start + 3 : pos_end]:
                            Ctot += phy_cap_Cmin2

                        v_cmaj_p[iter_ch - 1] = (
                            mdl["en_noi_cfb"] * np.random.randn() * np.sqrt(KT / Ctot)
                        )
                        v_cmaj_n[iter_ch - 1] = (
                            mdl["en_noi_cfb"] * np.random.randn() * np.sqrt(KT / Ctot)
                        )

                        if "1" in conf_col[pos_start + 3 : pos_end]:
                            v_cmin1_p[iter_ch - 1] = v_cmaj_p[iter_ch - 1]
                            v_cmin1_n[iter_ch - 1] = v_cmaj_n[iter_ch - 1]
                        if "2" in conf_col[pos_start + 3 : pos_end]:
                            v_cmin2_p[iter_ch - 1] = v_cmaj_p[iter_ch - 1]
                            v_cmin2_n[iter_ch - 1] = v_cmaj_n[iter_ch - 1]

                elif (
                    iter_task == 2
                ):  # charge share -------------------------------------
                    reg_pattern = r"CHS\d+"
                    match = re.search(reg_pattern, conf_col)
                    if match:
                        pos_start, pos_end = match.span()

                        Ctot = phy_cap_Cmaj
                        charge_p = v_cmaj_p[iter_ch - 1] * phy_cap_Cmaj
                        charge_n = v_cmaj_n[iter_ch - 1] * phy_cap_Cmaj

                        if "1" in conf_col[pos_start + 3 : pos_end]:
                            Ctot += phy_cap_Cmin1
                            charge_p += v_cmin1_p[iter_ch - 1] * phy_cap_Cmin1
                            charge_n += v_cmin1_n[iter_ch - 1] * phy_cap_Cmin1

                        if "2" in conf_col[pos_start + 3 : pos_end]:
                            Ctot += phy_cap_Cmin2
                            charge_p += v_cmin2_p[iter_ch - 1] * phy_cap_Cmin2
                            charge_n += v_cmin2_n[iter_ch - 1] * phy_cap_Cmin2

                        v_cmaj_p[iter_ch - 1] = charge_p / Ctot + mdl[
                            "en_noi_cfb"
                        ] * np.random.randn() * np.sqrt(KT / Ctot)
                        v_cmaj_n[iter_ch - 1] = charge_n / Ctot + mdl[
                            "en_noi_cfb"
                        ] * np.random.randn() * np.sqrt(KT / Ctot)

                        if "1" in conf_col[pos_start + 3 : pos_end]:
                            v_cmin1_p[iter_ch - 1] = v_cmaj_p[iter_ch - 1]
                            v_cmin1_n[iter_ch - 1] = v_cmaj_n[iter_ch - 1]

                        if "2" in conf_col[pos_start + 3 : pos_end]:
                            v_cmin2_p[iter_ch - 1] = v_cmaj_p[iter_ch - 1]
                            v_cmin2_n[iter_ch - 1] = v_cmaj_n[iter_ch - 1]

                        if mdl["is_verbose"] >= 2:
                            delta_cmin1 = (
                                v_cmin1_p[iter_ch - 1] - v_cmin1_n[iter_ch - 1]
                            )
                            delta_cmin2 = (
                                v_cmin2_p[iter_ch - 1] - v_cmin2_n[iter_ch - 1]
                            )
                            delta_cmaj = v_cmaj_p[iter_ch - 1] - v_cmaj_n[iter_ch - 1]

                            print(
                                f"[CHS][{iter_ch-1}] v_cmin1_p={v_cmin1_p[iter_ch-1] * 1e3:.4f} mV, "
                                f"v_cmin1_n={v_cmin1_n[iter_ch-1] * 1e3:.4f} mV, "
                                f"delta_cmin1={delta_cmin1 * 1e3:.4f} mV"
                            )

                            print(
                                f"[CHS][{iter_ch-1}] v_cmin2_p={v_cmin2_p[iter_ch-1] * 1e3:.4f} mV, "
                                f"v_cmin2_n={v_cmin2_n[iter_ch-1] * 1e3:.4f} mV, "
                                f"delta_cmin2={delta_cmin2 * 1e3:.4f} mV"
                            )

                            print(
                                f"[CHS][{iter_ch-1}] v_cmaj_p={v_cmaj_p[iter_ch-1] * 1e3:.4f} mV, "
                                f"v_cmaj_n={v_cmaj_n[iter_ch-1] * 1e3:.4f} mV, "
                                f"delta_cmaj={delta_cmaj * 1e3:.4f} mV"
                            )

                elif iter_task == 3:  # sample ---------------------------------
                    reg_pattern = r"SAM\d*F?"
                    match = re.search(reg_pattern, conf_col)

                    if match:
                        pos_start, pos_end = match.start(), match.end()

                        if conf_col[pos_start + 3] == "0":
                            v_cdac_p[iter_ch - 1] = 0
                            v_cdac_n[iter_ch - 1] = 0
                        elif conf_col[pos_start + 3] == "1":
                            v_cdac_p[iter_ch - 1] = v_in_p[0, iter_sim - 1]
                            v_cdac_n[iter_ch - 1] = v_in_n[0, iter_sim - 1]
                        elif conf_col[pos_start + 3] == "2":
                            v_cdac_p[iter_ch - 1] = v_in_p[1, iter_sim - 1]
                            v_cdac_n[iter_ch - 1] = v_in_n[1, iter_sim - 1]
                        else:
                            v_cdac_p[iter_ch - 1] = v_in_p[0, iter_sim - 1]
                            v_cdac_n[iter_ch - 1] = v_in_n[0, iter_sim - 1]

                        # flip chopping
                        if conf_col[pos_end - 1] == "F":
                            v_cdac_p[iter_ch - 1], v_cdac_n[iter_ch - 1] = (
                                v_cdac_n[iter_ch - 1],
                                v_cdac_p[iter_ch - 1],
                            )

                        # add noise
                        noise_factor = (
                            mdl["en_noi_cdac"]
                            * np.random.randn()
                            * np.sqrt(KT / phy_cap_CDAC)
                        )
                        v_cdac_p[iter_ch - 1] += noise_factor
                        v_cdac_n[iter_ch - 1] += noise_factor

                        # debug info
                        if mdl["is_verbose"] > 1:
                            delta_vcdac = (
                                v_cdac_p[iter_ch - 1] - v_cdac_n[iter_ch - 1]
                            ) * 1e3
                            print(
                                f"[SAM][{iter_ch-1}][{conf_col[pos_start:pos_end]}] v_cdac_p={v_cdac_p[iter_ch-1]*1e3:.4f} mV, "
                                f"v_cdac_n={v_cdac_n[iter_ch-1]*1e3:.4f} mV, Vcdac={delta_vcdac:.4f} mV"
                            )

                elif iter_task == 4:  # conversion---------------------------------
                    # MSB
                    if "MSB" in conf_col:
                        for iter_sar in range(n_bit_sar1):
                            # calculate comparator voltage
                            v_cmp[iter_ch - 1] = (
                                v_cdac_p[iter_ch - 1] - v_cdac_n[iter_ch - 1]
                            ) + (v_cmaj_p[iter_ch - 1] - v_cmaj_n[iter_ch - 1])
                            vn_cmp = (
                                mdl["en_noi_cmp"]
                                * np.random.randn()
                                * sgm_vn_cmp[iter_ch - 1]
                            )

                            if mdl["is_verbose"] >= 2:
                                delta_CDAC = (
                                    v_cdac_p[iter_ch - 1] - v_cdac_n[iter_ch - 1]
                                )
                                print(
                                    f"[MSB][{iter_ch-1}][iter_SAR={iter_sar + 1}] v_cdac_p={v_cdac_p[iter_ch-1] * 1e3:.4f} mV, "
                                    f"v_cdac_n={v_cdac_n[iter_ch-1] * 1e3:.4f} mV, delta_CDAC={delta_CDAC * 1e3:.4f} mV, "
                                    f"Vcmp={v_cmp[iter_ch-1] * 1e3:.4f} mV"
                                )

                            if v_cmp[iter_ch - 1] + v_cmp_os[iter_ch - 1] + vn_cmp > 0:
                                v_cdac_p[iter_ch - 1] -= (
                                    mdl["v_ref"] * a_wgt_sar1[iter_sar] / 2
                                )
                                v_cdac_n[iter_ch - 1] += (
                                    mdl["v_ref"] * a_wgt_sar1[iter_sar] / 2
                                )
                                d_out_ch[iter_ch - 1, iter_sar] = 1
                            else:
                                v_cdac_p[iter_ch - 1] += (
                                    mdl["v_ref"] * a_wgt_sar1[iter_sar] / 2
                                )
                                v_cdac_n[iter_ch - 1] -= (
                                    mdl["v_ref"] * a_wgt_sar1[iter_sar] / 2
                                )

                        rdym_ch[iter_ch - 1] = 1

                        if mdl["is_verbose"] >= 2:
                            print(
                                f"[MSB][{iter_ch-1}] v_cdac_p={v_cdac_p[iter_ch-1] * 1e3:.4f} mV, v_cdac_n={v_cdac_n[iter_ch-1] * 1e3:.4f} mV, Vcmp={v_cmp[iter_ch-1] * 1e3:.4f} mV"
                            )
                            print(f"[MSB][{iter_ch-1}] digital code =")
                            print(d_out_ch)

                    # LSB
                    if "LSB" in conf_col:
                        for iter_sar in range(n_bit_sar2):
                            # calculate comparator voltage
                            v_cmp[iter_ch - 1] = (
                                v_cdac_p[iter_ch - 1] - v_cdac_n[iter_ch - 1]
                            ) + (v_cmaj_p[iter_ch - 1] - v_cmaj_n[iter_ch - 1])
                            vn_cmp = (
                                mdl["en_noi_cmp"]
                                * np.random.randn()
                                * sgm_vn_cmp[iter_ch - 1]
                            )

                            if mdl["is_verbose"] >= 2:
                                delta_CDAC = (
                                    v_cdac_p[iter_ch - 1] - v_cdac_n[iter_ch - 1]
                                )
                                print(
                                    f"[LSB][{iter_ch-1}][iter_SAR={iter_sar + 1}] v_cdac_p={v_cdac_p[iter_ch-1] * 1e3:.4f} mV, "
                                    f"v_cdac_n={v_cdac_n[iter_ch-1] * 1e3:.4f} mV, delta_CDAC={delta_CDAC * 1e3:.4f} mV, "
                                    f"Vcmp={v_cmp[iter_ch-1] * 1e3:.4f} mV"
                                )

                            if v_cmp[iter_ch - 1] + v_cmp_os[iter_ch - 1] + vn_cmp > 0:
                                d_out_ch[iter_ch - 1, n_bit_sar1 + iter_sar] = 1
                                v_cdac_p[iter_ch - 1] -= (
                                    mdl["v_ref"] * a_wgt_sar2[iter_sar] / 2
                                )
                                v_cdac_n[iter_ch - 1] += (
                                    mdl["v_ref"] * a_wgt_sar2[iter_sar] / 2
                                )
                            else:
                                v_cdac_p[iter_ch - 1] += (
                                    mdl["v_ref"] * a_wgt_sar2[iter_sar] / 2
                                )
                                v_cdac_n[iter_ch - 1] -= (
                                    mdl["v_ref"] * a_wgt_sar2[iter_sar] / 2
                                )

                        rdyl_ch[iter_ch - 1] = 1

                        if mdl["is_verbose"] >= 2:
                            print(
                                f"[LSB][{iter_ch-1}] v_cdac_p={v_cdac_p[iter_ch-1] * 1e3:.4f} mV, v_cdac_n={v_cdac_n[iter_ch-1] * 1e3:.4f} mV, Vcmp={v_cmp[iter_ch-1] * 1e3:.4f} mV"
                            )
                            print(f"[LSB][{iter_ch-1}] digital code =")
                            print(d_out_ch)

                elif (
                    iter_task == 5
                ):  # amplification (bus TX) ---------------------------
                    reg_pattern = r"AMP\dF?"
                    match = re.search(reg_pattern, conf_col)

                    if match:
                        v_cmp[iter_ch - 1] = (
                            v_cdac_p[iter_ch - 1] - v_cdac_n[iter_ch - 1]
                        ) + (v_cmaj_p[iter_ch - 1] - v_cmaj_n[iter_ch - 1])

                        vres_p = v_cdac_p[iter_ch - 1] + v_cmaj_p[iter_ch - 1]
                        vres_n = v_cdac_n[iter_ch - 1] + v_cmaj_n[iter_ch - 1]

                        v_amp_diff = (
                            8 * (vres_p - vres_n)
                            + a_nl2_amp[iter_ch - 1] * (vres_p - vres_n) ** 2
                            + a_nl3_amp[iter_ch - 1] * (vres_p - vres_n) ** 3
                        ) / 8

                        i_amp_p = (v_amp_diff / 2 + v_amp_os[iter_ch - 1] / 2) * mdl[
                            "nominal_gm_amp"
                        ]
                        i_amp_n = (-v_amp_diff / 2 + v_amp_os[iter_ch - 1] / 2) * mdl[
                            "nominal_gm_amp"
                        ]

                        if mdl["is_verbose"] >= 2:
                            AMP_input = vres_p - vres_n
                            delta_CDAC = v_cdac_p[iter_ch - 1] - v_cdac_n[iter_ch - 1]
                            delta_CMAJ = v_cmaj_p[iter_ch - 1] - v_cmaj_n[iter_ch - 1]

                            print(
                                f"[AMP][{iter_ch-1}] v_cdac_p={v_cdac_p[iter_ch-1]*1e3:.4f} mV, v_cdac_n={v_cdac_n[iter_ch-1]*1e3:.4f} mV, delta_CDAC={delta_CDAC*1e3:.4f} mV"
                            )
                            print(
                                f"[AMP][{iter_ch-1}] v_cmaj_p={v_cmaj_p[iter_ch-1]*1e3:.4f} mV, v_cmaj_n={v_cmaj_n[iter_ch-1]*1e3:.4f} mV, delta_CMAJ={delta_CMAJ*1e3:.4f} mV"
                            )
                            print(
                                f"[AMP][{iter_ch-1}] AMP_input={AMP_input*1e3:.4f} mV, AMP_target={AMP_input*8*1e3:.4f} mV"
                            )

                        # gm chopping
                        if conf_col[match.end() - 1] == "F":
                            i_amp_p, i_amp_n = i_amp_n, i_amp_p

                        ## maybe problematic
                        bus_index = int(conf_col[match.start() + 3])
                        if bus_index == 0:
                            i_self_p[iter_ch - 1] = i_amp_p
                            i_self_n[iter_ch - 1] = i_amp_n
                        else:
                            i_bus_p[bus_index] = i_amp_p
                            i_bus_n[bus_index] = i_amp_n

                elif iter_task == 6:  # residue feedback sampling (bus RX)
                    reg_pattern = r"F\d[PN]\d+L?"
                    match = re.search(reg_pattern, conf_col)

                    if match:
                        pos_start = match.start()
                        pos_end = match.end()

                        # residue feedback path
                        bus_num = int(conf_col[pos_start + 1])
                        if bus_num == 0:
                            v_fb_p = i_self_p[iter_ch - 1] * mdl["nominal_TIA_gain"]
                            v_fb_n = i_self_n[iter_ch - 1] * mdl["nominal_TIA_gain"]
                        else:
                            v_fb_p = i_bus_p[bus_num] * mdl["nominal_TIA_gain"]
                            v_fb_n = i_bus_n[bus_num] * mdl["nominal_TIA_gain"]

                        # direction
                        if conf_col[pos_start + 2] == "P":
                            v_fb_p = v_fb_p * 1
                            v_fb_n = v_fb_n * 1
                        elif conf_col[pos_start + 2] == "N":
                            v_fb_p = v_fb_p * -1
                            v_fb_n = v_fb_n * -1

                        # output referred noise (including 2-phase)
                        if conf_col[pos_end - 1] == "L":
                            v_fb_p_noise = v_fb_p + mdl[
                                "en_noi_amp"
                            ] * np.random.randn() * sgm_vn_amp_lbw[iter_ch - 1] * mdl[
                                "nominal_gm_amp"
                            ] * mdl[
                                "nominal_TIA_gain"
                            ] / np.sqrt(
                                2
                            )
                            v_fb_n_noise = v_fb_n + mdl[
                                "en_noi_amp"
                            ] * np.random.randn() * sgm_vn_amp_lbw[iter_ch - 1] * mdl[
                                "nominal_gm_amp"
                            ] * mdl[
                                "nominal_TIA_gain"
                            ] / np.sqrt(
                                2
                            )
                        else:
                            v_fb_p_noise = (
                                v_fb_p
                                + mdl["en_noi_amp"]
                                * np.random.randn()
                                * sgm_vn_amp[iter_ch - 1]
                                * mdl["nominal_gm_amp"]
                                * mdl["nominal_TIA_gain"]
                            )
                            v_fb_n_noise = (
                                v_fb_n
                                + mdl["en_noi_amp"]
                                * np.random.randn()
                                * sgm_vn_amp[iter_ch - 1]
                                * mdl["nominal_gm_amp"]
                                * mdl["nominal_TIA_gain"]
                            )

                        Ctot = 0
                        if "0" in conf_col[pos_start + 3 : pos_end]:
                            Ctot += phy_cap_Cmaj
                        if "1" in conf_col[pos_start + 3 : pos_end]:
                            Ctot += phy_cap_Cmin1
                        if "2" in conf_col[pos_start + 3 : pos_end]:
                            Ctot += phy_cap_Cmin2

                        v_fb_p_noise += (
                            mdl["en_noi_cfb"] * np.random.randn() * np.sqrt(KT / Ctot)
                        )
                        v_fb_n_noise += (
                            mdl["en_noi_cfb"] * np.random.randn() * np.sqrt(KT / Ctot)
                        )

                        # update cap value
                        if "0" in conf_col[pos_start + 3 : pos_end]:
                            v_cmaj_p[iter_ch - 1] = v_fb_p_noise
                            v_cmaj_n[iter_ch - 1] = v_fb_n_noise
                        if "1" in conf_col[pos_start + 3 : pos_end]:
                            v_cmin1_p[iter_ch - 1] = v_fb_p_noise
                            v_cmin1_n[iter_ch - 1] = v_fb_n_noise
                        if "2" in conf_col[pos_start + 3 : pos_end]:
                            v_cmin2_p[iter_ch - 1] = v_fb_p_noise
                            v_cmin2_n[iter_ch - 1] = v_fb_n_noise

                        if mdl["is_verbose"] >= 2:
                            delta_cmin1 = (
                                v_cmin1_p[iter_ch - 1] - v_cmin1_n[iter_ch - 1]
                            )
                            delta_cmin2 = (
                                v_cmin2_p[iter_ch - 1] - v_cmin2_n[iter_ch - 1]
                            )
                            delta_cmaj = v_cmaj_p[iter_ch - 1] - v_cmaj_n[iter_ch - 1]

                            print(
                                f"[F][{iter_ch-1}] v_cmin1_p={v_cmin1_p[iter_ch-1]*1e3:.4f} mV, v_cmin1_n={v_cmin1_n[iter_ch-1]*1e3:.4f} mV, delta_cmin1={delta_cmin1*1e3:.4f} mV"
                            )
                            print(
                                f"[F][{iter_ch-1}] v_cmin2_p={v_cmin2_p[iter_ch-1]*1e3:.4f} mV, v_cmin2_n={v_cmin2_n[iter_ch-1]*1e3:.4f} mV, delta_cmin2={delta_cmin2*1e3:.4f} mV"
                            )
                            print(
                                f"[F][{iter_ch-1}] v_cmaj_p={v_cmaj_p[iter_ch-1]*1e3:.4f} mV, v_cmaj_n={v_cmaj_n[iter_ch-1]*1e3:.4f} mV, delta_cmaj={delta_cmaj*1e3:.4f} mV"
                            )

        d_out = np.array(d_out_list)
    return d_out
