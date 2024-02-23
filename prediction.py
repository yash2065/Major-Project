import cv2
import typing
import numpy as np
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder
import Levenshtein


def match_label(predicted_word, label_list):
    best_match = None
    min_distance = float('inf')

    for label in label_list:
        distance = Levenshtein.distance(predicted_word, label)

        if distance < min_distance:
            min_distance = distance
            best_match = label

    return best_match


class Model_run(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def recognise(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shape[:2][::-1])
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)
        preds = self.model.run(None, {self.input_name: image_pred})[0]
        predicted = ctc_decoder(preds, self.char_list)[0]

        return predicted


if __name__ == "__main__":
    import pandas as pd
    from mltu.configs import BaseModelConfigs

    configurations = BaseModelConfigs.load(
        "Models\\dr_recognition\\model2/configs.yaml")

    model = Model_run(
        model_path=configurations.model_path, char_list=configurations.vocab)

    df = pd.read_csv(
        "Models\\dr_recognition\\model2/val.csv").values.tolist()

    for image_path, label in df:
        image = cv2.imread(image_path)

        prediction = model.recognise(image)

        label_list = ["Brufen", "Crocin", "Claritin", "Aspirin", "Omez", "pantocid", "glycomet", "Amoxil",
                      "suprax", "azee", "levoquin", "ciplox", "Domstat", "zantac", "Volini", "ultram", "flagyi",
                      "Lipitor", "zocor", "rosuvas", "Norvasc", "cozaar", "altace", "Lopressor", "Vasotec", "micardis",
                      "singulair", "Ventolin", "Zoloft", "Prozac", "Lexapro", "paxil", "reglan", "allegra", "Zyrtec",
                      "xyzal", "Medrol", "deltasone", "Humulin", "amaryl", "Janumet", "actos", "rabicip",
                      "Nexium", "prevacid", "Pepcid", "duphalac", "Dulcolax", "Zofran", "patocid", "combiflam",
                      "ultracet", "glimet", "mifegyne", "oxytocin", "primolut", "ovral", "cholecalciferol",
                      "entamizole", "Aldactone", "Monistat", "difflucan", "augmentin", "paracetamol"]

        print(
            f"\nPrediction: {prediction}, \t Medicine:{match_label(prediction,label_list)}")

        image = cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2))
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
