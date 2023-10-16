# PleaseBlink! <image src="./PleaseBlink/icon.png" width="35"/>

<image src="./docs/image1.png" width="200"/>

Introducing PleaseBlink!—a lightweight application crafted to gently remind you to blink your eyes during prolonged screen exposure.

## Core Features

- **Blink Detection**: Harnessing the power of `dlib` for accurate blink recognition. Support both local and remote detection.
- **Interactive Notifications**: Utilizing `rumps` to provide user-friendly notifications.
- **Personalized Settings**: Easily adjustable detection and notification parameters to cater to your preferences.



## Getting Started

1. Use prebuilt `.app`
   Just download the latest release from [here]() and run it.

2. Use source code
  Clone this repo and install the dependencies as instructions on [Develop](#develop).
  Then change to root directory of this repo and run

   ```bash
   python ./PleaseBlink/main.py
   ```

   Use local blink detection by default. If you want to use remote detection, please run

   ```bash
   python ./PleaseBlink/main.py --detect_mode remote --remote_host <ip>:<port> # on the client side
   python ./PleaseBlink/server.py --port <port> # on the server side
   ```


Once running, PleaseBlink! operates quietly in the background monitoring your blink frequency.

Locate the icon on the status bar, click it to unveil the menu, and tweak the parameters as needed. The status bar icon will morph to `RED` indicating a prolonged period of no blinking. Upon detecting a blink, it shifts to `GREEN`.

PleaseBlink! will count the number of blinks over a specified duration. Should the blink count fall below the set threshold, a friendly reminder pops up prompting you to blink.

<image src="./docs/image2.png" width="400px">

> **Note❗️**: For the notification feature to function correctly, please ensure that Python has the necessary permissions to send notifications. Navigate to `System Preferences` → `Notifications`, scroll down to `PleaseBlink`, and activate the notification option.

## Develop

`PleaseBlink!` is tailored for MacOS environments.

- Execute `pip install -r requirements.txt` to install the necessary dependencies.
- Execute `pip install -r requirements_server.txt` to install dependencies on the server if you want to use remote detection.

> **Note❗**: It's noted that `rumps` and `dlib` might encounter issues with Anaconda's Python version, especially on M1 chips. It's recommended to use Brew's Python version and initiate a virtual environment.