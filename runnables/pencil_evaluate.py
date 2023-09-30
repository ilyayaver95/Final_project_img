from omegaconf import DictConfig
import hydra


from tasks.predict_task import Predict


config_path = r'C:\Users\IlyaY\PycharmProjects\Final_project_img\configs'  # configuration path
config_name = r'config.yaml'  # configuration name


@hydra.main(version_base=None, config_path=config_path, config_name=config_name)
def my_app(cfg: DictConfig):
    predicter = Predict(cfg)  # Initiate object of prediction
    try:
        result = predicter.run()  # run method that classifies the pencil.
    except:
        print('Bad example')  # in case that one of the processes crashed, print error.
        result = 'Error'

    print(result)


if __name__ == "__main__":
    my_app()  # run the app