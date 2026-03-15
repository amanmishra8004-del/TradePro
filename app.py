"""
Trade Edge – Flask Application
Trading Analysis, Risk Management & Backtesting Platform
"""

import io
import json
import csv
import os

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from werkzeug.utils import secure_filename
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    flash,
    session,
    Response,
    send_file,
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from datetime import datetime

# ── Local modules ──────────────────────────────────────────────────────────────
from strategies.sma_strategy import generate_signals as sma_signals
from strategies.rsi_strategy import generate_signals as rsi_signals
from backtest.engine import run_backtest
from risk.risk_manager import calculate_position

# ── App setup ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.secret_key = "trade_edge_secret_key_2026"
app.config["SQLALCHEMY_DATABASE_URI"] = (
    f"sqlite:///{os.path.join(BASE_DIR, 'database', 'db.sqlite3')}"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"] = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message_category = "info"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ── Database Models ────────────────────────────────────────────────────────────
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    profile_image = db.Column(db.String(120), nullable=True, default='default.png')
    strategies = db.relationship('Strategy', backref='author', lazy=True)
    backtests = db.relationship('BacktestResult', backref='author', lazy=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class Strategy(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(120), nullable=False)
    strategy_type = db.Column(db.String(50), nullable=False)  # sma | rsi | custom
    params = db.Column(db.Text, nullable=False)  # JSON string
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "strategy_type": self.strategy_type,
            "params": json.loads(self.params),
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M"),
        }


class BacktestResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    strategy_id = db.Column(db.Integer, db.ForeignKey("strategy.id"), nullable=True)
    symbol = db.Column(db.String(20), nullable=False)
    period = db.Column(db.String(20))
    metrics = db.Column(db.Text)  # JSON
    trades = db.Column(db.Text)  # JSON
    equity_curve = db.Column(db.Text)  # JSON
    run_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "symbol": self.symbol,
            "period": self.period,
            "metrics": json.loads(self.metrics) if self.metrics else {},
            "run_at": self.run_at.strftime("%Y-%m-%d %H:%M"),
        }


class ChartAnnotation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    symbol = db.Column(db.String(20), nullable=False)
    name = db.Column(db.String(100), default="Technical Analysis")
    shapes = db.Column(db.Text)  # JSON string
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "symbol": self.symbol,
            "name": self.name,
            "shapes": json.loads(self.shapes) if self.shapes else [],
            "updated_at": self.updated_at.strftime("%Y-%m-%d %H:%M")
        }

# ── Helpers ────────────────────────────────────────────────────────────────────
def _fetch_data(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize(None)
    return df


def _add_indicators(df: pd.DataFrame, indicators: list) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]

    for ind in indicators:
        if ind == "SMA20":
            df["SMA20"] = close.rolling(20).mean()
        elif ind == "SMA50":
            df["SMA50"] = close.rolling(50).mean()
        elif ind == "SMA200":
            df["SMA200"] = close.rolling(200).mean()
        elif ind == "EMA20":
            df["EMA20"] = close.ewm(span=20, adjust=False).mean()
        elif ind == "EMA50":
            df["EMA50"] = close.ewm(span=50, adjust=False).mean()
        elif ind == "RSI":
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = (-delta).clip(lower=0)
            avg_gain = gain.ewm(com=13, min_periods=14).mean()
            avg_loss = loss.ewm(com=13, min_periods=14).mean()
            rs = avg_gain / avg_loss
            df["RSI"] = 100 - (100 / (1 + rs))
        elif ind == "MACD":
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            df["MACD"] = ema12 - ema26
            df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
            df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
        elif ind == "BB":
            sma20 = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            df["BB_Upper"] = sma20 + 2 * std20
            df["BB_Mid"] = sma20
            df["BB_Lower"] = sma20 - 2 * std20
    return df


def _build_market_chart(df: pd.DataFrame, symbol: str, indicators: list) -> str:
    has_rsi = "RSI" in df.columns
    has_macd = "MACD" in df.columns
    n_rows = 1 + int(has_rsi) + int(has_macd)
    row_heights = [0.6] + [0.2] * (n_rows - 1)
    subplot_titles = [symbol]
    if has_rsi:
        subplot_titles.append("RSI (14)")
    if has_macd:
        subplot_titles.append("MACD")

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
        vertical_spacing=0.05,
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color="#00d4aa",
            decreasing_line_color="#ff4d6d",
        ),
        row=1, col=1,
    )

    # Overlay indicators
    colors = {
        "SMA20": "#f7b731", "SMA50": "#fd9644", "SMA200": "#a55eea",
        "EMA20": "#45aaf2", "EMA50": "#2bcbba",
        "BB_Upper": "#78909c", "BB_Mid": "#9e9e9e", "BB_Lower": "#78909c",
    }
    for col in ["SMA20", "SMA50", "SMA200", "EMA20", "EMA50"]:
        if col in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[col], name=col,
                           line=dict(color=colors[col], width=1.5)),
                row=1, col=1,
            )
    for bb_col in ["BB_Upper", "BB_Mid", "BB_Lower"]:
        if bb_col in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[bb_col], name=bb_col,
                           line=dict(color=colors[bb_col], width=1, dash="dash"),
                           showlegend=(bb_col == "BB_Upper")),
                row=1, col=1,
            )

    current_row = 2
    if has_rsi:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                       line=dict(color="#ffd32a", width=1.5)),
            row=current_row, col=1,
        )
        fig.add_hline(y=70, line_dash="dot", line_color="#ff4d6d", row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#00d4aa", row=current_row, col=1)
        current_row += 1

    if has_macd:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["MACD"], name="MACD",
                       line=dict(color="#45aaf2", width=1.5)),
            row=current_row, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal",
                       line=dict(color="#fd9644", width=1.5)),
            row=current_row, col=1,
        )
        fig.add_trace(
            go.Bar(x=df.index, y=df["MACD_Hist"], name="Histogram",
                   marker_color=["#00d4aa" if v >= 0 else "#ff4d6d"
                                 for v in df["MACD_Hist"].fillna(0)]),
            row=current_row, col=1,
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(family="Inter, sans-serif", color="#c9d1d9"),
        height=600 + 150 * (n_rows - 1),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(gridcolor="#21262d", showgrid=True)
    fig.update_yaxes(gridcolor="#21262d", showgrid=True)

    return fig.to_json()


def _build_equity_chart(equity_curve: list, initial_capital: float) -> str:
    dates = [e["date"] for e in equity_curve]
    values = [e["equity"] for e in equity_curve]
    pct = [(v / initial_capital - 1) * 100 for v in values]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates, y=values,
            name="Portfolio Value",
            fill="tozeroy",
            fillcolor="rgba(0,212,170,0.08)",
            line=dict(color="#00d4aa", width=2),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(family="Inter, sans-serif", color="#c9d1d9"),
        height=350,
        margin=dict(l=10, r=10, t=10, b=40),
        xaxis_gridcolor="#21262d",
        yaxis_gridcolor="#21262d",
    )
    return fig.to_json()


def _build_trade_dist_chart(trades: list) -> str:
    if not trades:
        return "{}"
    pnls = [t["pnl"] for t in trades]
    colors = ["#00d4aa" if p > 0 else "#ff4d6d" for p in pnls]
    fig = go.Figure(
        go.Bar(
            x=list(range(1, len(pnls) + 1)),
            y=pnls,
            marker_color=colors,
            name="Trade P&L",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(family="Inter, sans-serif", color="#c9d1d9"),
        height=280,
        margin=dict(l=10, r=10, t=10, b=40),
        xaxis=dict(title="Trade #", gridcolor="#21262d"),
        yaxis=dict(title="P&L ($)", gridcolor="#21262d"),
    )
    return fig.to_json()


# ── Routes ─────────────────────────────────────────────────────────────────────


# ── Authentication Routes ──────────────────────────────────────────────────────

@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        
        if not username or not email or not password:
            flash("All fields are required.", "warning")
            return redirect(url_for('register'))
            
        if User.query.filter_by(username=username).first():
            flash("Username already exists.", "error")
            return redirect(url_for('register'))
            
        if User.query.filter_by(email=email).first():
            flash("Email already registered.", "error")
            return redirect(url_for('register'))
            
        hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(username=username, email=email, password=hashed_pw)
        db.session.add(user)
        db.session.commit()
        flash("Registration successful! You can now log in.", "success")
        return redirect(url_for('login'))
        
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        
        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            flash(f"Welcome back, {user.username}!", "success")
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid credentials. Please try again.", "error")
            
    return render_template("login.html")


@app.route("/logout")
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))


# ── Protected Routes ───────────────────────────────────────────────────────────


@app.route("/")
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template("landing.html")

@app.route("/dashboard")
@login_required
def dashboard():
    recent = BacktestResult.query.filter_by(user_id=current_user.id).order_by(BacktestResult.run_at.desc()).limit(5).all()
    strategies = Strategy.query.filter_by(user_id=current_user.id).order_by(Strategy.created_at.desc()).limit(5).all()
    stats = {
        "total_backtests": BacktestResult.query.filter_by(user_id=current_user.id).count(),
        "total_strategies": Strategy.query.filter_by(user_id=current_user.id).count(),
    }
    return render_template(
        "dashboard.html",
        recent_results=[r.to_dict() for r in recent],
        strategies=[s.to_dict() for s in strategies],
        stats=stats,
    )


@app.route("/market")
@login_required
def market():
    return render_template("market_data.html")


@app.route("/strategy")
@login_required
def strategy_builder():
    strategies = Strategy.query.filter_by(user_id=current_user.id).order_by(Strategy.created_at.desc()).all()
    return render_template("strategy.html", strategies=[s.to_dict() for s in strategies])


@app.route("/backtesting")
@login_required
def backtesting():
    strategies = Strategy.query.filter_by(user_id=current_user.id).all()
    results = BacktestResult.query.filter_by(user_id=current_user.id).order_by(BacktestResult.run_at.desc()).limit(10).all()
    return render_template(
        "backtesting.html",
        strategies=[s.to_dict() for s in strategies],
        results=[r.to_dict() for r in results],
    )


@app.route("/risk")
@login_required
def risk_management():
    return render_template("risk.html")


@app.route("/reports")
@login_required
def reports():
    results = BacktestResult.query.filter_by(user_id=current_user.id).order_by(BacktestResult.run_at.desc()).all()
    return render_template("reports.html", results=[r.to_dict() for r in results])


@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    if request.method == "POST":
        update_type = request.form.get("update_type", "image")
        
        if update_type == "image":
            if 'profile_image' in request.files:
                file = request.files['profile_image']
                if file and file.filename != '' and allowed_file(file.filename):
                    filename = secure_filename(f"{current_user.id}_{int(datetime.now().timestamp())}_{file.filename}")
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                    current_user.profile_image = filename
                    db.session.commit()
                    flash("Profile image updated successfully!", "success")
                else:
                    flash("Invalid file type or no file selected.", "error")
                    
        elif update_type == "email":
            new_email = request.form.get("email", "").strip()
            if new_email:
                existing = User.query.filter_by(email=new_email).first()
                if existing and existing.id != current_user.id:
                    flash("Email is already in use by another account.", "error")
                else:
                    current_user.email = new_email
                    db.session.commit()
                    flash("Email updated successfully!", "success")
            else:
                flash("Email cannot be empty.", "warning")
                
        elif update_type == "password":
            new_pw = request.form.get("new_password", "")
            if len(new_pw) >= 6:
                hashed_pw = bcrypt.generate_password_hash(new_pw).decode('utf-8')
                current_user.password = hashed_pw
                db.session.commit()
                flash("Password updated successfully!", "success")
            else:
                flash("Password must be at least 6 characters long.", "warning")

        return redirect(url_for("settings"))
    return render_template("settings.html")


@app.route("/admin")
@login_required
def admin_panel():
    if current_user.username != "shivam":
        flash("Unauthorized access. Admins only.", "error")
        return redirect(url_for("dashboard"))
    
    users = User.query.all()
    strategies = Strategy.query.all()
    results = BacktestResult.query.all()
    
    return render_template("admin.html", users=users, strategies=strategies, results=results)

@app.route("/admin/download_db")
@login_required
def download_db():
    if current_user.username != "shivam":
        return redirect(url_for("dashboard"))
    db_path = app.config["SQLALCHEMY_DATABASE_URI"].replace("sqlite:///", "")
    if os.path.exists(db_path):
        return send_file(db_path, as_attachment=True, download_name="trade_edge_db.sqlite3")
    flash("Database file not found.", "error")
    return redirect(url_for("admin_panel"))

@app.route("/admin/download_excel")
@login_required
def download_excel():
    if current_user.username != "shivam":
        return redirect(url_for("dashboard"))
    
    try:
        import pandas as pd
        from io import BytesIO
        
        # 1. Users Data - Safe Extraction
        users = User.query.all()
        user_list = []
        for u in users:
            user_list.append({
                "ID": u.id,
                "Username": getattr(u, 'username', 'N/A'),
                "Email": getattr(u, 'email', 'N/A'),
                "Total Strategies": len(u.strategies) if u.strategies else 0,
                "Total Backtests": len(u.backtests) if u.backtests else 0
            })
        users_df = pd.DataFrame(user_list)
        
        # 2. Strategies Data - Safe Extraction
        strategies = Strategy.query.all()
        strat_list = []
        for s in strategies:
            strat_list.append({
                "ID": s.id,
                "Author": s.author.username if s.author else "System",
                "Name": s.name,
                "Type": s.strategy_type,
                "Created At": s.created_at.strftime("%Y-%m-%d %H:%M") if s.created_at else "N/A"
            })
        strat_df = pd.DataFrame(strat_list)
        
        # 3. Backtest Results Data - Enhanced & Robust
        results = BacktestResult.query.all()
        backtest_list = []
        for r in results:
            try:
                m = json.loads(r.metrics) if r.metrics else {}
            except:
                m = {}
            
            backtest_list.append({
                "ID": r.id,
                "Author": r.author.username if r.author else "System",
                "Symbol": r.symbol,
                "Period": r.period if r.period else "N/A",
                "Win Rate": f"{m.get('win_rate', 0):.1f}%",
                "Return (%)": f"{m.get('total_return_pct', 0):+.2f}%",
                "Final Capital": m.get('final_capital', 0),
                "Total Trades": m.get('total_trades', 0),
                "Max Drawdown": f"{m.get('max_drawdown_pct', 0):.2f}%",
                "Run At": r.run_at.strftime("%Y-%m-%d %H:%M") if r.run_at else "N/A"
            })
        results_df = pd.DataFrame(backtest_list)
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            users_df.to_excel(writer, index=False, sheet_name='All Users')
            strat_df.to_excel(writer, index=False, sheet_name='All Strategies')
            results_df.to_excel(writer, index=False, sheet_name='Backtest History')
            
            # Auto-adjust columns width
            for sheetname in writer.sheets:
                worksheet = writer.sheets[sheetname]
                for col in worksheet.columns:
                    max_length = 0
                    column = col[0].column_letter
                    for cell in col:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except: pass
                    worksheet.column_dimensions[column].width = max_length + 2

        output.seek(0)
        return send_file(
            output,
            as_attachment=True,
            download_name=f"TradeEdge_Master_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        flash(f"Excel Export Failed: {str(e)}", "error")
        return redirect(url_for("admin_panel"))


# ── API Endpoints ──────────────────────────────────────────────────────────────


@app.route("/api/market/search", methods=["GET"])
@login_required
def api_market_search():
    query = request.args.get("q", "")
    if not query or len(query) < 1:
        return jsonify([])
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=10&newsCount=0"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=5).json()
        quotes = res.get('quotes', [])
        results = []
        for q in quotes:
            if 'symbol' in q and 'shortname' in q:
                results.append({
                    "symbol": q['symbol'],
                    "name": q['shortname'],
                    "exchange": q.get('exchDisp', ''),
                    "type": q.get('quoteType', '')
                })
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/market/recommendations", methods=["GET"])
@login_required
def api_market_recommendations():
    category = request.args.get("category", "indian")
    
    # Define stock lists per category
    assets = {
        "indian": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "AXISBANK.NS", "LT.NS", "KOTAKBANK.NS"],
        "us": ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "NFLX", "AMD", "ADBE"],
        "crypto": ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "ADA-USD", "AVAX-USD", "DOT-USD", "TRX-USD"],
        "forex": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X", "USDCAD=X", "USDINR=X", "EURINR=X", "GBPINR=X"],
        "it": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "LTIM.NS", "TECHM.NS", "AAPL", "MSFT", "NVDA", "GOOGL"],
        "ev": ["TATAMOTORS.NS", "M&M.NS", "ONGC.NS", "TSLA", "RIVN", "LCID", "NIO", "XPEV", "LI", "BYDDF"],
        "health": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS", "JNJ", "PFE", "UNH", "MRK", "ABT"],
        "auto": ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "HEROMOTOCO.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "F", "GM", "TM", "HMC"],
        "edu": ["ZEELEARN.NS", "APTECHT.NS", "NIITLTD.NS", "CHGG", "COUR", "DUOL", "UDMY", "TAL", "EDU", "BYJU.BO"],
        "gainers": ["NVDA", "TSLA", "META", "AMZN", "AAPL", "MSFT", "TCS.NS", "RELIANCE.NS", "BTC-USD", "ETH-USD", "TATAMOTORS.NS", "RELIANCE.NS"],
        "losers": ["NVDA", "TSLA", "META", "AMZN", "AAPL", "MSFT", "TCS.NS", "RELIANCE.NS", "BTC-USD", "ETH-USD", "TATAMOTORS.NS", "RELIANCE.NS"]
    }
    
    symbols_to_fetch = assets.get(category, assets["indian"])
    results = []
    
    try:
        # Increased period to 10d and use dropna to handle weekends/holidays
        data = yf.download(symbols_to_fetch, period="10d", interval="1d", group_by="ticker", progress=False)
        for sym in symbols_to_fetch:
            if len(symbols_to_fetch) == 1:
                df = data
            else:
                df = data[sym] if sym in data else pd.DataFrame()
            
            # Clean NaNs - crucial for weekend data
            df = df.dropna(subset=["Close"])
            
            if not df.empty and len(df) >= 2:
                last_close = float(df["Close"].iloc[-1])
                prev_close = float(df["Close"].iloc[-2])
                
                if pd.isna(last_close) or pd.isna(prev_close):
                    continue

                change = last_close - prev_close
                change_pct = (change / prev_close) * 100
                
                results.append({
                    "symbol": sym.replace(".NS", "").replace("=X", ""),
                    "raw_symbol": sym,
                    "price": round(last_close, 2),
                    "change": round(change, 2),
                    "change_pct": round(change_pct, 2)
                })
        if category == "gainers":
            results.sort(key=lambda x: x["change_pct"], reverse=True)
            results = [r for r in results if r["change_pct"] > 0]
        elif category == "losers":
            results.sort(key=lambda x: x["change_pct"])
            results = [r for r in results if r["change_pct"] < 0]
        else:
            results.sort(key=lambda x: x["change_pct"], reverse=True)
            
        return jsonify({"category": category, "assets": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/market/annotations", methods=["GET", "POST", "DELETE"])
@login_required
def api_market_annotations():
    if request.method == "POST":
        data = request.get_json()
        symbol = data.get("symbol", "").upper().strip()
        shapes = data.get("shapes", [])
        name = data.get("name", "Technical Analysis")
        
        if not symbol:
            return jsonify({"error": "Symbol is required"}), 400
            
        annotation = ChartAnnotation.query.filter_by(user_id=current_user.id, symbol=symbol).first()
        if not annotation:
            annotation = ChartAnnotation(user_id=current_user.id, symbol=symbol)
            db.session.add(annotation)
            
        annotation.shapes = json.dumps(shapes)
        annotation.name = name
        db.session.commit()
        return jsonify({"message": "Technical Analysis saved"})
    
    elif request.method == "DELETE":
        symbol = request.args.get("symbol", "").upper().strip()
        annotation = ChartAnnotation.query.filter_by(user_id=current_user.id, symbol=symbol).first()
        if annotation:
            db.session.delete(annotation)
            db.session.commit()
            return jsonify({"message": "Analysis deleted"})
        return jsonify({"error": "Not found"}), 404
        
    else:
        symbol = request.args.get("symbol", "").upper().strip()
        annotation = ChartAnnotation.query.filter_by(user_id=current_user.id, symbol=symbol).first()
        return jsonify(annotation.to_dict() if annotation else {"symbol": symbol, "shapes": [], "name": ""})


@app.route("/api/market/recent-analysis")
@login_required
def api_market_recent_analysis():
    # Only return symbols where user has drawn something
    recent = ChartAnnotation.query.filter_by(user_id=current_user.id).order_by(ChartAnnotation.updated_at.desc()).limit(10).all()
    return jsonify([r.to_dict() for r in recent])


@app.route("/api/market-data", methods=["POST"])
@login_required
def api_market_data():
    data = request.get_json()
    symbol = data.get("symbol", "AAPL").upper().strip()
    period = data.get("period", "1y")
    interval = data.get("interval", "1d")
    indicators = data.get("indicators", [])

    try:
        df = _fetch_data(symbol, period, interval)
        if df.empty:
            return jsonify({"error": f"No data found for {symbol}"}), 404

        df = _add_indicators(df, indicators)

        # Latest quote info
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
        change = float(last["Close"] - prev["Close"])
        change_pct = float(change / prev["Close"] * 100)

        info = {
            "symbol": symbol,
            "last_close": round(float(last["Close"]), 4),
            "open": round(float(last["Open"]), 4),
            "high": round(float(last["High"]), 4),
            "low": round(float(last["Low"]), 4),
            "volume": int(last["Volume"]) if not pd.isna(last["Volume"]) else 0,
            "change": round(change, 4),
            "change_pct": round(change_pct, 3),
        }

        chart_json = _build_market_chart(df, symbol, indicators)
        return jsonify({"info": info, "chart": chart_json})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/strategy/save", methods=["POST"])
@login_required
def api_save_strategy():
    data = request.get_json()
    name = data.get("name", "").strip()
    strategy_type = data.get("strategy_type", "sma")
    params = data.get("params", {})

    if not name:
        return jsonify({"error": "Strategy name is required"}), 400

    # Check duplicate name
    existing = Strategy.query.filter_by(name=name, user_id=current_user.id).first()
    if existing:
        # Update
        existing.strategy_type = strategy_type
        existing.params = json.dumps(params)
        db.session.commit()
        return jsonify({"message": "Strategy updated", "id": existing.id})

    s = Strategy(name=name, strategy_type=strategy_type, params=json.dumps(params), user_id=current_user.id)
    db.session.add(s)
    db.session.commit()
    return jsonify({"message": "Strategy saved", "id": s.id})


@app.route("/api/strategy/delete/<int:sid>", methods=["DELETE"])
@login_required
def api_delete_strategy(sid):
    s = Strategy.query.filter_by(id=sid, user_id=current_user.id).first_or_404()
    db.session.delete(s)
    db.session.commit()
    return jsonify({"message": "Deleted"})


@app.route("/api/strategy/list")
@login_required
def api_list_strategies():
    strategies = Strategy.query.filter_by(user_id=current_user.id).order_by(Strategy.created_at.desc()).all()
    return jsonify([s.to_dict() for s in strategies])


@app.route("/api/backtest/run", methods=["POST"])
@login_required
def api_run_backtest():
    data = request.get_json()
    symbol = data.get("symbol", "AAPL").upper().strip()
    period = data.get("period", "2y")
    interval = data.get("interval", "1d")
    strategy_type = data.get("strategy_type", "sma")
    params = data.get("params", {})
    capital = float(data.get("capital", 100_000))
    risk_pct = float(data.get("risk_pct", 1.0))
    stop_loss_pct = float(data.get("stop_loss_pct", 2.0))
    take_profit_pct = float(data.get("take_profit_pct", 4.0))

    try:
        df = _fetch_data(symbol, period, interval)
        if df.empty:
            return jsonify({"error": f"No data for {symbol}"}), 404

        # Generate signals
        if strategy_type == "sma":
            fast = int(params.get("fast", 50))
            slow = int(params.get("slow", 200))
            df = sma_signals(df, fast=fast, slow=slow)
        elif strategy_type == "rsi":
            rsi_period = int(params.get("rsi_period", 14))
            oversold = float(params.get("oversold", 30))
            overbought = float(params.get("overbought", 70))
            df = rsi_signals(df, period=rsi_period, oversold=oversold, overbought=overbought)
        else:
            return jsonify({"error": "Unknown strategy type"}), 400

        result = run_backtest(
            df,
            initial_capital=capital,
            risk_pct=risk_pct,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )

        # Save result to DB
        bt = BacktestResult(
            user_id=current_user.id,
            symbol=symbol,
            period=period,
            metrics=json.dumps(result["metrics"]),
            trades=json.dumps(result["trades"]),
            equity_curve=json.dumps(result["equity_curve"]),
        )
        db.session.add(bt)
        db.session.commit()

        equity_chart = _build_equity_chart(result["equity_curve"], capital)
        trade_dist_chart = _build_trade_dist_chart(result["trades"])

        return jsonify(
            {
                "metrics": result["metrics"],
                "trades": result["trades"][-50:],  # last 50 trades
                "equity_chart": equity_chart,
                "trade_dist_chart": trade_dist_chart,
                "result_id": bt.id,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/risk/calculate", methods=["POST"])
@login_required
def api_risk_calculate():
    data = request.get_json()
    try:
        result = calculate_position(
            capital=float(data.get("capital", 100_000)),
            risk_pct=float(data.get("risk_pct", 1.0)),
            entry_price=float(data.get("entry_price", 100)),
            stop_loss_pct=float(data.get("stop_loss_pct", 2.0)),
            take_profit_pct=float(data.get("take_profit_pct", 4.0)),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/reports/export/<int:result_id>")
@login_required
def api_export_csv(result_id):
    bt = BacktestResult.query.filter_by(id=result_id, user_id=current_user.id).first_or_404()
    trades = json.loads(bt.trades)
    metrics = json.loads(bt.metrics)
    return render_template("report_template.html", bt=bt, trades=trades, metrics=metrics)


@app.route("/api/backtest/result/<int:result_id>")
@login_required
def api_get_result(result_id):
    bt = BacktestResult.query.filter_by(id=result_id, user_id=current_user.id).first_or_404()
    trades = json.loads(bt.trades)
    equity_curve = json.loads(bt.equity_curve)
    metrics = json.loads(bt.metrics)
    capital_used = metrics.get("final_capital", 100_000) - metrics.get("total_pnl", 0)
    equity_chart = _build_equity_chart(equity_curve, capital_used)
    trade_dist_chart = _build_trade_dist_chart(trades)
    return jsonify({
        "metrics": metrics,
        "trades": trades,
        "equity_chart": equity_chart,
        "trade_dist_chart": trade_dist_chart,
        "symbol": bt.symbol,
        "period": bt.period,
    })


# ── App init ───────────────────────────────────────────────────────────────────
with app.app_context():
    db.create_all()
    
    # Create admin user if not exists
    admin_user = User.query.filter_by(username="shivam").first()
    if not admin_user:
        hashed_pw = bcrypt.generate_password_hash("shivam123").decode('utf-8')
        admin_user = User(username="shivam", email="shivam@tradeedge.com", password=hashed_pw)
        db.session.add(admin_user)
        db.session.commit()

if __name__ == "__main__":
    app.run(debug=True, port=5000)

