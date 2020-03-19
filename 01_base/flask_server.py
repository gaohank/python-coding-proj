from flask import Flask, request, json, session, make_response, abort, render_template, redirect, url_for
from re import escape

flask_test = Flask(__name__)
flask_test.config["SECRET_KEY"] = "123456"


@flask_test.route('/abort')
def abort():
    abort(404)


@flask_test.errorhandler(404)
def page_not_found(error):
    resp = make_response(render_template('error.html'), 404)
    resp.headers['X-Something'] = 'A value'
    return resp


@flask_test.route('/')
def index():
    if 'username' in session:
        return 'Logged in as %s' % escape(session['username'])
    return 'You are not logged in'


@flask_test.route('/json')
def json():
    return {"name": 'hank', "age": 30}


@flask_test.route('/user/<username>')
def show_user_profile(username):
    # show the user profile for that user
    return 'User %s' % escape(username)


@flask_test.route('/post/<int:post_id>')
def show_post(post_id):
    # show the post with the given id, the id is an integer
    return 'Post %d' % post_id


@flask_test.route('/path/<path:subpath>')
def show_subpath(subpath):
    # show the subpath after /path/
    return 'Subpath %s' % escape(subpath)


@flask_test.route('/about')
def about():
    return 'The about page'


@flask_test.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        session['username'] = request.form['username']
        return redirect(url_for('about'))
    return '''
        <form method="post">
            <p><input type=text name=username>
            <p><input type=submit value=Login>
        </form>
    '''


@flask_test.route('/logout')
def logout():
    # remove the username from the session if it's there
    session.pop('username', None)
    return redirect(url_for('index'))


if __name__ == '__main__':
    flask_test.run(host='0.0.0.0', port=7800)
