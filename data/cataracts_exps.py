EXP1 = {"LABEL": {
    0: [0],
    1: [1],
    2: [2],
    3: [3],
    4: [4],
    5: [5],
    6: [6],
    7: [7, 8, 9, 10, 11,
        12, 13, 14, 15, 16,
        17, 18, 19, 20, 21,
        22, 23, 24, 25, 26,
        27, 28, 29, 30, 31,
        32, 33, 34, 35]
},
    "CLASS": {
        0: "Pupil",
        1: "Surgical Tape",
        2: "Hand",
        3: "Eye Retractors",
        4: "Iris",
        5: "Skin",
        6: "Cornea",
        7: "Instrument"
    }
}

EXP2 = {"LABEL": {
    0: [0],
    1: [1],
    2: [2],
    3: [3],
    4: [4],
    5: [5],
    6: [6],
    7: [7, 8, 10, 27, 20, 32],
    8: [9, 22],
    9: [11, 33],
    10: [12, 28],
    11: [13, 21],
    12: [14, 24],
    13: [15, 18],
    14: [16, 23],
    15: [17],
    16: [19],
    255: [25, 26, 29, 30, 31, 34, 35],
},
    "CLASS": {
        0: "Pupil",
        1: "Surgical Tape",
        2: "Hand",
        3: "Eye Retractors",
        4: "Iris",
        5: "Skin",
        6: "Cornea",
        7: "Cannula",
        8: "Cap. Cystotome",
        9: "Tissue Forceps",
        10: "Primary Knife",
        11: "Ph. Handpiece",
        12: "Lens Injector",
        13: "I/A Handpiece",
        14: "Secondary Knife",
        15: "Micromanipulator",
        16: "Cap. Forceps",
        255: "Ignore",
    }
}

EXP3 = {"LABEL": {
    0: [0],
    1: [1],
    2: [2],
    3: [3],
    4: [4],
    5: [5],
    6: [6],
    7: [7],
    8: [8],
    9: [9],
    10: [10],
    11: [11],
    12: [12],
    13: [13],
    14: [14],
    15: [15],
    16: [16],
    17: [17],
    18: [18],
    19: [19],
    20: [20],
    21: [21],
    22: [22],
    23: [23],
    24: [24],
    255: [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
},
    "CLASS": {
        0: "Pupil",
        1: "Surgical Tape",
        2: "Hand",
        3: "Eye Retractors",
        4: "Iris",
        5: "Skin",
        6: "Cornea",
        7: "Hydro. Cannula",
        8: "Visc. Cannula",
        9: "Cap. Cystotome",
        10: "Rycroft Cannula",
        11: "Bonn Forceps",
        12: "Primary Knife",
        13: "Ph. Handpiece",
        14: "Lens Injector",
        15: "I/A Handpiece",
        16: "Secondary Knife",
        17: "Micromanipulator",
        18: "I/A Handpiece Handle",
        19: "Cap. Forceps",
        20: "R. Cannula Handle",
        21: "Ph. Handpiece Handle",
        22: "Cap. Cystotome Handle",
        23: "Sec. Knife Handle",
        24: "Lens Injector Handle",
        255: "Ignore",
    }
}

# https://arxiv.org/abs/2506.21813
CAT_SG_EXP = {
    "LABEL": {
        **{i: [i] for i in range(0, 18)},   # 0–17 → 0–17
        18: [19],                          # 19 → 18
        19: [25],                          # 25 → 19
        20: [26],                          # 26 → 20
        21: [27],                          # 27 → 21
        22: [29],                          # 29 → 22
        23: [30],                          # 30 → 23
        24: [31],                          # 31 → 24
        25: [33],                          # 33 → 25
        26: [34],                          # 34 → 26
        27: [35],                          # 35 → 27
        255: [18, 20, 21, 22, 23, 24, 28, 32, 255],
    },

    "CLASS": {
        0: "Pupil",
        1: "Surgical Tape",
        2: "Hand",
        3: "Eye Retractors",
        4: "Iris",
        5: "Skin",
        6: "Cornea",
        7: "Hydro. Cannula",
        8: "Visc. Cannula",
        9: "Cap. Cystotome",
        10: "Rycroft Cannula",
        11: "Bonn Forceps",
        12: "Primary Knife",
        13: "Ph. Handpiece",
        14: "Lens Injector",
        15: "I/A Handpiece",
        16: "Secondary Knife",
        17: "Micromanipulator",
        18: "Cap. Forceps",
        19: "Suture Needle",
        20: "Needle Holder",
        21: "Charleux Cannula",
        22: "Vitrectomy Handpiece",
        23: "Mendez Ring",
        24: "Marker",
        25: "Troutman Forceps",
        26: "Cotton",
        27: "Iris Hooks",
        255: "Ignore",
    }
}

def build_lut(exp_dict, max_label=256, ignore_index=255):
    label_lut = [ignore_index] * max_label

    for new_label, old_labels in exp_dict["LABEL"].items():
        for old in old_labels:
            label_lut[old] = new_label

    # explicitly allowed ignored labels
    intentional_ignore = set(exp_dict["LABEL"].get(ignore_index, []))

    # labels that appear in mapping (excluding ignore bucket)
    mentioned = set()
    for k, v in exp_dict["LABEL"].items():
        if k != ignore_index:
            mentioned.update(v)

    # labels that are mentioned but still end up ignored
    unexpected_ignore = sorted(
        l for l in mentioned
        if label_lut[l] == ignore_index and l not in intentional_ignore
    )

    if unexpected_ignore:
        raise ValueError(
            f"Mapped labels sent to ignore unexpectedly: {unexpected_ignore}"
        )

    return label_lut, exp_dict.get("CLASS", {})


