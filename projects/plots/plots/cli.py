import jsonargparse
from plots.main import main as vizapp


def main(args=None):
    parser = jsonargparse.ArgumentParser()
    parser.add_function_arguments(vizapp)
    args = parser.parse_args()
    vizapp(**vars(args))


if __name__ == "__main__":
    main()
