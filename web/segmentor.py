import torch
from model import GDLoss
from model import CELoss
from model import AttrDict
from model import Baseline_2
import torch.nn as nn
import numpy as np
import os
import pickle
import nibabel as nib
from utils import crop_image
from utils import preprocess_image
from utils import Patient
from utils import transform
from utils import postprocess_image

PATIENTS_PATH = "./uploads/preprocessed/"
POSTPROCESSED_PATH = "./uploads/postprocessed/"
spacing_target = [4.5250006, 1.1986301, 1.1986301]


class Segmentor:
    def __init__(self):
        if not os.path.isdir(PATIENTS_PATH):
            os.makedirs(PATIENTS_PATH)
        if not os.path.isdir(POSTPROCESSED_PATH):
            os.makedirs(POSTPROCESSED_PATH)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Model = Baseline_2
        self.model = nn.ModuleDict(
            [
                [axis, Model(AttrDict(**{"lr": 0.01, "functions": [GDLoss, CELoss]}))]
                for axis in ["SA", "LA"]
            ]
        ).to(self.device)

        for file in os.listdir("./models"):
            if not file.endswith(".pth"):
                continue
            axis = file[6:8]
            ckpt = torch.load(
                f"./models/model_{axis}.pth",
                map_location=torch.device("cpu"),
            )
            self.model[axis].load_state_dict(ckpt["M"])
            self.model[axis].optimizer.load_state_dict(ckpt["M_optim"])
            self.model[axis].eval()

    def generate_patient_info(self, filename, id, axis):
        if not os.path.exists(os.path.join(PATIENTS_PATH, "patient_info.pkl")):
            patient_info = {}
            with open(os.path.join(PATIENTS_PATH, "patient_info.pkl"), "wb") as f:
                pickle.dump(patient_info, f)
        with open(os.path.join(PATIENTS_PATH, "patient_info.pkl"), "rb") as f:
            patient_info = pickle.load(f)

        image_path = os.path.join(PATIENTS_PATH, filename)
        patient_info[id] = {}
        print(image_path)
        print(image_path.format("CINE"))
        image = nib.load(image_path.format("CINE"))
        patient_info[id]["spacing"] = image.header["pixdim"][[3, 2, 1]]
        patient_info[id]["header"] = image.header
        patient_info[id]["affine"] = image.affine
        patient_info[id]["path"] = os.path.join(PATIENTS_PATH, filename)

        image_ED = nib.load(image_path.format("ED")).get_fdata()
        image_ES = nib.load(image_path.format("ES")).get_fdata()

        patient_info[id]["shape_ED"] = image_ED.shape
        patient_info[id]["shape_ES"] = image_ES.shape
        patient_info[id]["crop_ED"] = crop_image(image_ED)
        patient_info[id]["crop_ES"] = crop_image(image_ES)
        with open(os.path.join(PATIENTS_PATH, "patient_info.pkl"), "wb") as f:
            pickle.dump(patient_info, f)
        self.preprocess(patient_info[id], id, axis)

    def preprocess(self, patient_info, id, axis):
        image_path = patient_info["path"]
        images = {}
        for j in ["ED", "ES"]:
            fname = image_path.format(j)
            if not os.path.isfile(fname):
                continue
            image = preprocess_image(
                nib.load(fname).get_fdata(),
                patient_info["crop_{}".format(j)],
                patient_info["spacing"],
                spacing_target,
            )
            images[j] = image.astype(np.float32)
        patient = Patient(patient_info, images, id, transform)

        self.run_inference(patient, axis)

    def run_inference(self, patient, axis):
        prediction = []
        for iter, batch in enumerate(patient):
            if batch["data"].dim() == 3: 
                batch["data"] = batch["data"].unsqueeze(0)
            batch = {"data": batch["data"].to(self.device)}
            with torch.no_grad():
                batch["prediction"] = self.model[axis].forward(batch["data"])
            batch["prediction"] = batch["prediction"][0]
            prediction = (
                torch.cat([prediction, batch["prediction"]], dim=0)
                if len(prediction) > 0
                else batch["prediction"]
            )
        if len(prediction) != 0:
            prediction = {
                "ED": prediction[: len(prediction) // 2].cpu().numpy(),
                "ES": prediction[len(prediction) // 2 :].cpu().numpy(),
            }
        result = {"ED": prediction["ED"], "ES": prediction["ES"]}
        self.postprocess_predictions(patient, result)

    def postprocess_predictions(self, patient, result):
        results = {"ED": {}, "ES": {}}
        patient_id = patient.id
        for key, val in result.items():
            prediction = postprocess_image(val, patient.info, key, spacing_target)
            nib.save(
                nib.Nifti1Image(
                    prediction,
                    patient.info["affine"],
                    patient.info["header"],
                ),
                os.path.join(
                    POSTPROCESSED_PATH, "{}_{}.nii.gz".format(patient_id, key)
                ),
            )
        return results
