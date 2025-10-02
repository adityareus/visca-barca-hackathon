# LLama Daredevil

A vision assistor for visually impaired people powered by Cerebras and Meta llama-3.3-70b-vision model.

# Contents

<!-- - [Why?](#why) -->
- [Project Structure](#project-structure)
<!-- - [Prerequisites](#prerequisites)
- [Installation](#installation) -->
- [TODO](#todo)


### Project Structure
```shell
vision-navigator/
│
├── README.md                   # Project overview and quick start
├── requirements.txt            # Python dependencies
├── .env.example               # Example environment variables
├── .env                       # Your API key (create this)
├── .gitignore                 # Git ignore rules
├── setup.sh                   # Setup script for Mac/Linux
├── setup.bat                  # Setup script for Windows
│
├── src/                       # Source code
│   ├── __init__.py           # Package initialization
│   ├── main.py               # Main application entry point
│   ├── vision_analyzer.py    # Scene analysis with Cerebras
│   └── audio_feedback.py     # Text-to-speech module
│
├── config/                    # Configuration
│   └── settings.py           # Application settings
│
├── tests/                     # Test suite
│   └── test_basic.py         # Basic functionality tests
│
├── docs/                      # Documentation
│   └── usage.md              # Detailed usage guide
│
└── logs/                      # Application logs (auto-created)
```

### TODO

---

- [ ] Build application starting from scene analysis and text-to-speech module.