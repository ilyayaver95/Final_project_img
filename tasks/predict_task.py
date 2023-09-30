from skimage import io, color
from skimage.morphology import disk, diamond, square, ball  # noqa
import joblib
import warnings
from omegaconf import DictConfig


from tasks.utils_skimage import corrections, erosion_, dilation_, closing_, find_corner, create_polygon, angle_calc,\
    calculate_total_polygon_angle, create_new_polygon, show_and_compare_polygons, show_images


# Suppress all warnings
warnings.filterwarnings("ignore")


class Predict:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.image_path = self.cfg.image_path  # get the image path from configuration
        self.classifier_path = self.cfg.classifier_path  # get the model path from configuration

        self.smallest_angle_threshold = 17.6  # thresholds for manual classifying
        self.total_angle_threshold = 180
        self.area_diff_threshold = 0.1

    def predict(self, smallest_angle, total_angle, area_diff):

        loaded_classifier = joblib.load(self.classifier_path)  # Load the trained classifier from a file

        new_example = [[smallest_angle, total_angle, area_diff]]  # Example data for prediction (should have 3 features)

        # Make predictions on the new example
        predicted_class = loaded_classifier.predict(new_example)     # If it's a binary classification problem (0 or 1), predicted_class will contain the predicted class label (0 or 1).
                                                                     # predicted_proba will contain the probability of belonging to class 1.
        predicted_proba = loaded_classifier.predict_proba(new_example)

        print(f"Predicted Class: {predicted_class[0]}")

        return predicted_class[0]

    def enhance_picture(self):
        # Load the image
        image = io.imread(self.image_path)
        image = color.rgb2gray(image)

        logarithmic_corrected = corrections(image)  # make correction for black enhancement
        result_image = erosion_(logarithmic_corrected)  # get rid of noise and correct the edges
        result_image = dilation_(result_image)
        result_image = closing_(result_image)

        return result_image, image

    def get_features(self, result_image):
        pos = find_corner(result_image)  # find the corners of our object -  the tip of the pencil

        # Filter pairs based on the threshold
        pos = [pair for pair in pos if pair[0] >= 20 and pair[1] >= 20]  # sometimes the corners set to the picture corner, so we filter them
        # Filter pairs based on the threshold
        pos = [pair for pair in pos if pair[0] <= 230 and pair[1] <= 230]

        polygon = create_polygon(pos)  # create polygon based on the corners

        smallest_angle = angle_calc(polygon)  # calculate the feature
        print(f"The smallest angle in the polygon is {smallest_angle:.2f} degrees.")
        total_angle = calculate_total_polygon_angle(polygon)  # calculate the feature
        print(f"The total angle in the polygon is {total_angle:.2f} degrees.")
        new_polygon = create_new_polygon(polygon)  # create perfect polygon for comparison
        area_diff = show_and_compare_polygons(polygon, new_polygon)  # calculate the feature

        return smallest_angle, total_angle, area_diff

    def prediction_by_threshold(self, smallest_angle, total_angle, area_diff):
        smallest_angle_threshold = 3
        total_angle_threshold = 50
        area_diff_threshold = 2

        result = 0

        if smallest_angle < smallest_angle_threshold:
            result = result + 1
        if total_angle < total_angle_threshold:
            result = result + 1
        if area_diff < area_diff_threshold:
            result = result + 1

        if result >= 2:
            return 1
        else:
            return 0

    def run(self,):
        result_image, image = self.enhance_picture()  # preprocessing

        smallest_angle, total_angle, area_diff = self.get_features(result_image)  # fetch the features

        show_images(image, result_image)

        prediction = self.predict(smallest_angle, total_angle, area_diff)  # return the prediction of the model on example
        # prediction = self.prediction_by_threshold(smallest_angle, total_angle, area_diff)  # manual threshold

        if prediction == 1:  # translate the prediction to corresponding string
            return 'sharp'
        else:
            return 'broken'
