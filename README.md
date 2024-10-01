# Linköping University's Text Exploration Tool
Currently under review as a demo at EKAW 2024. See also https://github.com/LiUSemWeb/entity-recontextualization.

## Running the system
This is a fairly incomplete set of instructions to run the system; they will be improved shortly.
To run the system, for now you will need to have Flutter and Python installed.
After cloning, navigate to the root of the project.
Build the web project using `flutter build web --web-renderer html`.
This should add content to `lintext/build/web` including, among other things, `index.html`.
With this running, you can start a webserver using `python lintext/server/python/main.py`.
The UI can be access by default at `localhost:13679`.

You can also run the webserver without building the web version of the UI.
Then you need to open/run the UI through other means, such as VSCode by opening `main.dart` and building/running the app specifically for your OS.

## Requirements
TBD.
- Python 3.8+
- CUDA (Can probably work fine without, but will be very slow.)
- flutter 3.5.0+ (If building yourself)

See `lintext/pubspec.yaml` for Flutter dependencies and `lintext/server/python/requirements.txt` for Python dependencies.


