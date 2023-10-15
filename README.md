# PleaseBlink!

<image src="./docs/image1.png" width="200"/>

Introducing PleaseBlink!—a lightweight application crafted to gently remind you to blink your eyes during prolonged screen exposure.

## Core Features

- **Blink Detection**: Harnessing the power of `dlib` for accurate blink recognition.
- **Interactive Notifications**: Utilizing `rumps` to provide user-friendly notifications.
- **Personalized Settings**: Easily adjustable detection and notification parameters to cater to your preferences.

## Setting Up

`PleaseBlink!` is tailored for MacOS environments.

- Execute `pip install -r requirements.txt` to install the necessary dependencies.

> **Heads Up❗**: It's noted that `rumps` and `dlib` might encounter issues with Anaconda's Python version, especially on M1 chips. It's recommended to use Brew's Python version and initiate a virtual environment.

## Getting Started

- Launch the application using `python main.py`.
- Locate the icon on the status bar, click it to unveil the menu, and tweak the parameters as needed.

Once running, PleaseBlink! operates quietly in the background monitoring your blink frequency.

The status bar icon will morph to `RED` indicating a prolonged period of no blinking. Upon detecting a blink, it shifts to `GREEN`.

PleaseBlink! will count the number of blinks over a specified duration. Should the blink count fall below the set threshold, a friendly reminder pops up prompting you to blink.

<image src="./docs/image2.png" width="400px">

For the notification feature to function correctly, please ensure that Python has the necessary permissions to send notifications. Navigate to `System Preferences` → `Notifications`, scroll down to `Python`, and activate the notification option.