from setuptools import setup

APP = ["test.py"]  # Replace 'main.py' with the name of your main Python file
DATA_FILES = []  # Add any additional data files here, such as images or sounds
OPTIONS = {
    "argv_emulation": True,  # Allows the app to be run from Finder or the Dock
    "plist": {  # Customize the app's Info.plist file
        "CFBundleName": "My App",
        "CFBundleDisplayName": "My App",
        "CFBundleGetInfoString": "My App v1.0",
        "CFBundleIdentifier": "com.mycompany.myapp",
        "CFBundleVersion": "1.0",
        "CFBundleShortVersionString": "1.0",
        "CFBundleExecutable": "My App",
        "LSUIElement": True,  # Hides the app's Dock icon
    },
    "packages": ["pygame"],  # Add any additional packages here
    "includes": ["pygame.examples.aliens"],  # Add any additional modules here
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
