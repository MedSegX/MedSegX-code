dimension_dict = {
    "CBCT": "2D", 
    "Colon": "2D",
    "Colonos": "2D",
    "Colonoscopy": "2D",
    "CT": "3D",
    "CTA": "2D",
    "Dermos": "2D",
    "Dermoscopy": "2D",
    "Endos": "2D",
    "Endoscope": "2D",
    "Fundus": "2D",
    "Funduscopy": "2D",
    "MRI": "3D",
    "US": "2D",
    "xray": "2D",
    "Xray": "2D",
}

dimension_map = {
    "2D": 0,
    "3D": 1,
}

dimension_map_inv = {v: k for k, v in dimension_map.items()}
dimension_map_idx = {k: i for i, k in enumerate(dimension_map_inv.keys())}

modal_dict = {
    "CBCT": "CBCT", 
    "Colon": "Colonos",
    "Colonos": "Colonos",
    "Colonoscopy": "Colonos",
    "CT": "CT",
    "CTA": "CTA",
    "Dermos": "Dermos",
    "Dermoscopy": "Dermos",
    "Endos": "Endos",
    "Endoscope": "Endos",
    "Fundus": "Fundus",
    "Funduscopy": "Fundus",
    "MRI": "MRI",
    "US": "US",
    "xray": "Xray",
    "Xray": "Xray",
}

modal_map = {
    "CBCT": 0, 
    "Colonos": 1,
    "CT": 2,
    "CTA": 3,
    "Dermos": 4,
    "Endos": 5,
    "Fundus": 6,
    "MRI": 7,
    "US": 8,
    "Xray": 9,
}

modal_map_inv = {v: k for k, v in modal_map.items()}
modal_map_idx = {k: i for i, k in enumerate(modal_map_inv.keys())}

organ_level_1_dict = {
    "head&neck": ["LeftCaudatum", "RightCaudatum", "VestibularSchwannoma", "BrainWholeTumor", "BrainCoreTumor", 
                  "BrainEnhancingTumor", "IntracranialHemorrhage", "IschemicStrokeLesion", "MultipleSclerosis", "WhiteMatterHyperintensity", 
                  "WhiteMatter", "GrayMatter", "CerebroSpinalFluid", "Hippocampus", "Pituitary", 
                  "OpticCup", "OpticDisc", "RightInnerEar", "LeftMiddleEar", "GTVnx", 
                  "Tooth", "Lip", "CavityOral", "BuccalMucosa", "Glottis", "PharynxCancer", 
                  "GTVnd", "GlndThyroid", "ThyroidNodule", "ParotidL", "GindSubmandL"],
    "body": ["LeftLung", "RightLung", "LungCancer", "COVID", "GroundGlassesOpacities", 
             "Consolidation", "PleuraEffusion", "LungPneumothorax", "Bronchus", "PulmonaryAirways", 
             "Esophagus", "Heart", "LeftAtrium", "LVMyocardialEdema", "LVMyocardialScars", 
             "LeftAtrialBloodChamber", "RightAtrialBloodChamber", "LeftVentricle", "RightVentricle", "LeftVentricularBloodChamber", 
             "RightVentricularBloodChamber", "LeftVentricularMyocardium", "RightVentricularMyocardium", "NoReflow", 
             "MyocardialInfarction", "MitralValve", "LeftVentricleEpicardium", "BreastCancer", "Liver", 
             "LiverCancer", "GallBladder", "Pancreas", "PancreasCancer", "Spleen", 
             "LeftKidney", "RightKidney", "KidneyTumor", "LeftAdrenalGland", "RightAdrenalGland", 
             "Stomach", "StomachCancer", "Duodenum", "Colon", "ColonCancer", 
             "Rectum", "Intestine", "Polyp", "Instrument", "Bladder", 
             "Uterus", "Prostate", "ProstateCentralGland", "ProstatePeripheralZone", "ProstateTransitionalZone", 
             "ProstateTumor", "InterstitialThickening", "PulmonaryLibrosis", "FNH", "HCC", 
             "Hemangioma", "ICC"],
    "skeleton": ["Mandible", "CervicalSpine", "LeftCollarBone", "RightCollarBone", "CollarBone", 
             "ThoracicSpine", "LumbarSpine", "Sacrum", "LeftHip", "RightHip", 
             "FemurBone", "FemurCartilage", "TibiaBone", "TibiaCartilage"], 
    "vessel": ["RetinalVessel", "CarotidArteryRight", "CarotidVesselWall", "PulmonaryArtery", "PulmonaryEmbolism", 
               "Aorta", "AscendingAorta", "TL", "FL", "FLT", "Postcava"], 
    "skin": ["Skin"], 
}

organ_level_1_map = {
    "head&neck": 0,
    "body": 1,
    "skeleton": 2,
    "vessel": 3,
    "skin": 4,
}

organ_level_1_map_inv = {v: k for k, v in organ_level_1_map.items()}
organ_level_1_map_idx = {k: i for i, k in enumerate(organ_level_1_map_inv.keys())}

organ_level_2_dict = {
    "head": ["LeftCaudatum", "RightCaudatum", "VestibularSchwannoma", "BrainWholeTumor", "BrainCoreTumor", 
             "BrainEnhancingTumor", "IntracranialHemorrhage", "IschemicStrokeLesion", "MultipleSclerosis", "WhiteMatterHyperintensity", 
             "WhiteMatter", "GrayMatter", "CerebroSpinalFluid", "Hippocampus", "Pituitary"], 
    "face": ["OpticCup", "OpticDisc", "RightInnerEar", "LeftMiddleEar", "GTVnx", 
             "Tooth", "Lip", "CavityOral", "BuccalMucosa"], 
    "neck": ["Glottis", "PharynxCancer", "GTVnd", "GlndThyroid", "ThyroidNodule", 
             "ParotidL", "GindSubmandL"], 
    "chest": ["LeftLung", "RightLung", "LungCancer", "COVID", "GroundGlassesOpacities", 
              "Consolidation", "PleuraEffusion", "LungPneumothorax", "Bronchus", "PulmonaryAirways", 
              "Esophagus", "Heart", "LeftAtrium", "LVMyocardialEdema", "LVMyocardialScars", 
              "LeftAtrialBloodChamber", "RightAtrialBloodChamber", "LeftVentricle", "RightVentricle", "LeftVentricularBloodChamber", 
              "RightVentricularBloodChamber", "LeftVentricularMyocardium", "RightVentricularMyocardium", "NoReflow", 
              "MyocardialInfarction", "MitralValve", "LeftVentricleEpicardium", "BreastCancer", "InterstitialThickening", 
              "PulmonaryLibrosis"], 
    "abdomen": ["Liver", "LiverCancer", "GallBladder", "Pancreas", "PancreasCancer", 
                "Spleen", "LeftKidney", "RightKidney", "KidneyTumor", "LeftAdrenalGland", 
                "RightAdrenalGland", "Stomach", "StomachCancer", "Duodenum", "Colon", 
                "ColonCancer", "Rectum", "Intestine", "Polyp", "Instrument", 
                "FNH", "HCC", "Hemangioma", "ICC"], 
    "pelvis": ["Bladder", "Uterus", "Prostate", "ProstateCentralGland", "ProstatePeripheralZone", 
               "ProstateTransitionalZone", "ProstateTumor"], 
    "cranium&face": ["Mandible"],
    "cervix": ["CervicalSpine"], 
    "torso": ["LeftCollarBone", "RightCollarBone", "CollarBone", "ThoracicSpine", "LumbarSpine", "Sacrum"], 
    "limb": ["LeftHip", "RightHip", "FemurBone", "FemurCartilage", "TibiaBone", "TibiaCartilage"], 
    "vessel": ["RetinalVessel", "CarotidArteryRight", "CarotidVesselWall", "PulmonaryArtery", "PulmonaryEmbolism", 
               "Aorta", "AscendingAorta", "TL", "FL", "FLT", "Postcava"], 
    "skin": ["Skin"],
}

organ_level_2_map = {
    "head": 0,
    "face": 1,
    "neck": 2,
    "chest": 3,
    "abdomen": 4,
    "pelvis": 5,
    "cranium&face": 6,
    "cervix": 7,
    "torso": 8,
    "limb": 9,
    "vessel": 10,
    "skin": 11,
}

organ_level_2_map_inv = {v: k for k, v in organ_level_2_map.items()}
organ_level_2_map_idx = {k: i for i, k in enumerate(organ_level_2_map_inv.keys())}

organ_level_3_dict = {
    "brain": ["LeftCaudatum", "RightCaudatum", "VestibularSchwannoma", "BrainWholeTumor", "BrainCoreTumor", 
              "BrainEnhancingTumor", "IntracranialHemorrhage", "IschemicStrokeLesion", "MultipleSclerosis", "WhiteMatterHyperintensity", 
              "WhiteMatter", "GrayMatter", "CerebroSpinalFluid"], 
    "hippocampus": ["Hippocampus"], 
    "pituitary": ["Pituitary"],
    "eye": ["OpticCup", "OpticDisc"],
    "ear": ["RightInnerEar", "LeftMiddleEar"], 
    "nose": ["GTVnx"],
    "mouth": ["Tooth", "Lip", "CavityOral", "BuccalMucosa"],
    "throat": ["Glottis", "PharynxCancer"],
    "lymphnode": ["GTVnd"],
    "thyroid": ["GlndThyroid", "ThyroidNodule"],
    "parotid": ["ParotidL"],
    "saliarygland": ["GindSubmandL"],
    "lung": ["LeftLung", "RightLung", "LungCancer", "COVID", "GroundGlassesOpacities", 
             "Consolidation", "PleuraEffusion", "LungPneumothorax", "InterstitialThickening", "PulmonaryLibrosis"], 
    "trachea": ["Bronchus", "PulmonaryAirways"], 
    "esophagus": ["Esophagus"],
    "heart": ["Heart", "LeftAtrium", "LVMyocardialEdema", "LVMyocardialScars", "LeftAtrialBloodChamber", 
              "RightAtrialBloodChamber", "LeftVentricle", "RightVentricle", "LeftVentricularBloodChamber", 
              "RightVentricularBloodChamber", "LeftVentricularMyocardium", "RightVentricularMyocardium", "NoReflow", 
              "MyocardialInfarction", "MitrLeftVentricularMyocardium", "RightVentricularMyocardium", "NoReflow", 
              "MyocardialInfarction", "MitralValve", "LeftVentricleEpicardium"], 
    "breast": ["BreastCancer"],
    "liver": ["Liver", "LiverCancer", "FNH", "HCC", "Hemangioma", "ICC"],
    "gallbladder": ["GallBladder"],
    "pancreas": ["Pancreas", "PancreasCancer"],
    "spleen": ["Spleen"],
    "kidney": ["LeftKidney", "RightKidney", "KidneyTumor"], 
    "adrenal": ["LeftAdrenalGland", "RightAdrenalGland"], 
    "stomach": ["Stomach", "StomachCancer"],
    "intestine": ["Duodenum", "Colon", "ColonCancer", "Rectum", "Intestine", 
                  "Polyp", "Instrument"], 
    "bladder": ["Bladder"],
    "uterus": ["Uterus"],
    "prostate": ["Prostate", "ProstateCentralGland", "ProstatePeripheralZone", "ProstateTransitionalZone", "ProstateTumor"], 
    "jawbone": ["Mandible"],
    "cervicalspine": ["CervicalSpine"],
    "collarbone": ["LeftCollarBone", "RightCollarBone", "CollarBone"],
    "thoracicspine": ["ThoracicSpine"],
    "lumbarspine": ["LumbarSpine"],
    "sacrum": ["Sacrum"],
    "hip": ["LeftHip", "RightHip"],
    "femur": ["FemurBone", "FemurCartilage"],
    "tibia": ["TibiaBone", "TibiaCartilage"],
    "vessel": ["RetinalVessel", "CarotidArteryRight", "CarotidVesselWall", "PulmonaryArtery", "PulmonaryEmbolism", 
               "Aorta", "AscendingAorta", "TL", "FL", "FLT", "Postcava"], 
    "skin": ["Skin"],
}

organ_level_3_map = {
    "brain": 0,
    "hippocampus": 1,
    "pituitary": 2,
    "eye": 3,
    "ear": 4,
    "nose": 5,
    "mouth": 6,
    "throat": 7,
    "lymphnode": 8,
    "thyroid": 9,
    "parotid": 10,
    "saliarygland": 11,
    "lung": 12,
    "trachea": 13,
    "esophagus": 14,
    "heart": 15,
    "breast": 16,
    "liver": 17,
    "gallbladder": 18,
    "pancreas": 19,
    "spleen": 20,
    "kidney": 21,
    "adrenal": 22,
    "stomach": 23,
    "intestine": 24,
    "bladder": 25,
    "uterus": 26,
    "prostate": 27,
    "jawbone": 28,
    "cervicalspine": 29,
    "thoracicspine": 30,
    "lumbarspine": 31,
    "collarbone": 32,
    "sacrum": 33,
    "hip": 34,
    "femur": 35,
    "tibia": 36,
    "vessel": 37,
    "skin": 38,
}

organ_level_3_map_inv = {v: k for k, v in organ_level_3_map.items()}
organ_level_3_map_idx = {k: i for i, k in enumerate(organ_level_3_map_inv.keys())}

task_list = [
    "LeftCaudatum", "RightCaudatum", "VestibularSchwannoma", "BrainWholeTumor", "BrainCoreTumor", 
    "BrainEnhancingTumor", "IntracranialHemorrhage", "IschemicStrokeLesion", "MultipleSclerosis", "WhiteMatterHyperintensity", 
    "WhiteMatter", "GrayMatter", "CerebroSpinalFluid", "Hippocampus", "Pituitary", 
    "OpticCup", "OpticDisc", "RightInnerEar", "LeftMiddleEar", "GTVnx", 
    "Tooth", "Lip", "CavityOral", "BuccalMucosa", "Glottis", 
    "PharynxCancer", "GTVnd", "GlndThyroid", "ThyroidNodule", "ParotidL", 
    "GindSubmandL", "LeftLung", "RightLung", "LungCancer", "COVID", 
    "GroundGlassesOpacities", "Consolidation", "PleuraEffusion", "LungPneumothorax", "Bronchus", 
    "PulmonaryAirways", "Esophagus", "Heart", "LeftAtrium", "LVMyocardialEdema", 
    "LVMyocardialScars", "LeftAtrialBloodChamber", "RightAtrialBloodChamber", "LeftVentricle", "RightVentricle", 
    "LeftVentricularBloodChamber", "RightVentricularBloodChamber", "LeftVentricularMyocardium", "RightVentricularMyocardium", 
    "NoReflow", "MyocardialInfarction", "MitralValve", "LeftVentricleEpicardium", "BreastCancer", 
    "Liver", "LiverCancer", "GallBladder", "Pancreas", "PancreasCancer", 
    "Spleen", "LeftKidney", "RightKidney", "KidneyTumor", "LeftAdrenalGland", 
    "RightAdrenalGland", "Stomach", "Duodenum", "Colon", "ColonCancer", 
    "Rectum", "Intestine", "Polyp", "Instrument", "Bladder", 
    "Uterus", "Prostate", "ProstateCentralGland", "ProstatePeripheralZone", "ProstateTransitionalZone", 
    "ProstateTumor", "Mandible", "CervicalSpine", "LeftCollarBone", "RightCollarBone", 
    "CollarBone", "ThoracicSpine", "LumbarSpine", "Sacrum", "LeftHip", 
    "RightHip", "FemurBone", "FemurCartilage", "TibiaBone", "TibiaCartilage", 
    "RetinalVessel", "CarotidArteryRight", "CarotidVesselWall", "PulmonaryArtery", "PulmonaryEmbolism", 
    "Aorta", "AscendingAorta", "TL", "FL", "FLT", 
    "Postcava", "Skin"
]
task_idx = {k: i for i, k in enumerate(task_list)}


if __name__ == "__main__":
    print("modal number:", len(modal_map))
    print("organ level 1 number:", len(organ_level_1_map))
    print("organ level 2 number:", len(organ_level_2_map))
    print("organ level 3 number:", len(organ_level_3_map))
    print("task number:", len(task_list))
    