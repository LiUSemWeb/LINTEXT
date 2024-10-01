# Linköping University's Text Exploration Tool
Currently under review as a demo at EKAW 2024. See also https://github.com/LiUSemWeb/entity-recontextualization.

## Running the system
To run the system, for now you will need to have Flutter and Python installed.
After cloning, navigate to the root of the project.
Build the web project using `flutter build web --web-renderer html`.
This should add content to `lintext/build/web` including, among other things, `index.html`.
With this running, you can start a webserver using `python lintext/server/python/main.py`.
The UI can be access by default at `localhost:13679`.

You can also run the webserver without building the web version of the UI.
Then you need to open/run the UI through other means, such as VSCode by opening `main.dart` and building/running the app specifically for your OS.

## Environment
You likely want to run the python side of things in a virtual environment.
If you're using conda, you can use the `.yml` file provided to create the environment we use: `conda env create -f lintext/server/relex.yml`.

### Python requirements
- Python 3.8
- uvicorn 0.29
- fastapi 0.115
- pytorch 2.4.1
- pytorch-cuda 12.4
- transformers 4.40.2

### Flutter
See `lintext/pubspec.yaml` for Flutter dependencies. We use flutter sdk version 3.5.0.
