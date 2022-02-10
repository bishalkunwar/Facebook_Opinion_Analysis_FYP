from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


if __name__ == '__main__':
    app.run()


# Script for virtual env: virtualenv --system-site-packages -p python3 ./venv 
# and then : 

