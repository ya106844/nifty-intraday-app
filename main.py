import textwrap

def main():
    main_py = textwrap.dedent(r"""
        import flask

        app = flask.Flask(__name__)

        @app.route('/')
        def home():
            return "Nifty Intraday App is Running!"

        if __name__ == '__main__':
            app.run(host='0.0.0.0', port=5000)
    """)

    print("App code generated.")
    with open("app.py", "w") as f:
        f.write(main_py)

    print("Starting Flask server...")
    import os
    os.system("python app.py")

if __name__ == '__main__':
    main()
