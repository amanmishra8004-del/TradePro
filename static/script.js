/**
 * Trade Edge – Global JavaScript Utilities
 */

/* ── Toast Notifications ──────────────────────────────── */
(function createToastContainer() {
  const container = document.createElement('div');
  container.id = 'toastContainer';
  document.body.appendChild(container);
})();

function showToast(message, type = 'info', duration = 3500) {
  const icons = {
    success: 'fa-check-circle', error: 'fa-circle-xmark',
    warning: 'fa-triangle-exclamation', info: 'fa-circle-info'
  };
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `<i class="fas ${icons[type] || icons.info}"></i><span>${message}</span>`;
  document.getElementById('toastContainer').appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transition = 'opacity 0.3s';
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

/* ── Loading Overlay ──────────────────────────────────── */
function showLoading(text = 'Loading...') {
  document.getElementById('loadingText').textContent = text;
  document.getElementById('loadingOverlay').classList.add('active');
}
function hideLoading() {
  document.getElementById('loadingOverlay').classList.remove('active');
}

/* ── Currency ─────────────────────────────────────────── */
const CURRENCY_MAP = {
  'USD': { symbol: '$', rate: 1 },
  'INR': { symbol: '₹', rate: 83.5 }, // Approx
  'EUR': { symbol: '€', rate: 0.92 },
  'GBP': { symbol: '£', rate: 0.79 }
};

let currentCurrency = localStorage.getItem('tradeEdgeCurrency') || 'USD';

function updateGlobalCurrency(cc) {
  currentCurrency = cc;
  localStorage.setItem('tradeEdgeCurrency', cc);
  // Reload ticker and any visible prices
  loadTicker();
  if (typeof fetchMarketData === 'function') fetchMarketData();
  showToast(`Currency switched to ${cc}`, 'success');
}

function formatPrice(val) {
  const c = CURRENCY_MAP[currentCurrency];
  const converted = val * c.rate;
  return c.symbol + converted.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

/* ── Sidebar Toggle ───────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  const toggle = document.getElementById('sidebarToggle');
  if (toggle) {
    toggle.addEventListener('click', () => {
      document.getElementById('sidebar').classList.toggle('collapsed');
      document.getElementById('mainWrapper').classList.toggle('sidebar-collapsed');
    });
  }

  // Clock
  updateClock();
  setInterval(updateClock, 1000);

  // Market status
  updateMarketStatus();
});

function updateClock() {
  const el = document.getElementById('sidebarTime');
  if (!el) return;
  const now = new Date();
  el.textContent = now.toLocaleTimeString('en-US', { hour12: false });
}

function updateMarketStatus() {
  const dot = document.getElementById('marketStatusDot');
  const text = document.getElementById('marketStatusText');
  if (!dot || !text) return;
  const now = new Date();
  const day = now.getDay();         // 0=Sun, 6=Sat
  const hour = now.getUTCHours();
  const min  = now.getUTCMinutes();
  // NYSE: Mon-Fri 14:30–21:00 UTC
  const isWeekday = day >= 1 && day <= 5;
  const totalMin = hour * 60 + min;
  const isOpen = isWeekday && totalMin >= 870 && totalMin < 1260;
  dot.style.background = isOpen ? '#00d4aa' : '#ff4d6d';
  text.textContent = isOpen ? 'Market Open' : 'Market Closed';
}

/* ── Ticker tape ──────────────────────────────────────── */
const TICKER_SYMBOLS = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'BTC-USD', 'ETH-USD', 'GC=F', 'EURUSD=X'];
const tickerData = [];
let chartShapeHistory = []; // For Undo functionality

async function loadTicker() {
  const tape = document.getElementById('tickerTape');
  if (!tape) return;
  try {
    const results = await Promise.allSettled(
      TICKER_SYMBOLS.map(sym =>
        fetch('/api/market-data', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ symbol: sym, period: '5d', interval: '1d' })
        }).then(res => {
          if (!res.ok) throw new Error('Fail');
          return res.json();
        })
      )
    );
    
    // Clear old data
    tickerData.length = 0;
    results.forEach(r => {
      if (r.status === 'fulfilled' && r.value && r.value.info) {
        tickerData.push(r.value.info);
      }
    });

    if (tickerData.length > 0) {
      renderTicker();
    } else {
      tape.innerHTML = '<div class="ticker-item loading" style="color:var(--accent-orange)">Market Feed Offline</div>';
    }
  } catch (e) {
    tape.innerHTML = '<div class="ticker-item loading" style="color:var(--accent-red)">Connection Error</div>';
  }
}

// History for Undo
function pushChartHistory(shapes) {
  chartShapeHistory.push(JSON.parse(JSON.stringify(shapes)));
  if (chartShapeHistory.length > 20) chartShapeHistory.shift();
}

function renderTicker() {
  const tape = document.getElementById('tickerTape');
  if (!tape || tickerData.length === 0) return;

  tape.innerHTML = tickerData.map(item => {
    const isUp = item.change_pct >= 0;
    const icon = isUp ? 'fa-arrow-trend-up' : 'fa-arrow-trend-down';
    const colorClass = isUp ? 'text-green' : 'text-red';
    
    return `
      <div class="ticker-item">
        <span class="ticker-symbol">${item.symbol}</span>
        <span class="ticker-price">${formatPrice(item.last_close)}</span>
        <span class="ticker-change ${colorClass}">
          <i class="fas ${icon}"></i> ${Math.abs(item.change_pct).toFixed(2)}%
        </span>
      </div>
    `;
  }).join('');
}

// Global click handler for dropdowns
document.addEventListener('click', (e) => {
  const dropdowns = document.querySelectorAll('.profile-dropdown');
  dropdowns.forEach(dd => {
    if (!dd.contains(e.target)) {
      dd.classList.remove('active');
    }
  });
});

// Load ticker on page load – runs on all pages
document.addEventListener('DOMContentLoaded', () => {
    // Sync selector
    const sel = document.getElementById('currencySelector');
    if(sel) sel.value = currentCurrency;
    
    // initSidebar(); // Assuming this function exists elsewhere or is meant to be added
    loadTicker();
    setInterval(loadTicker, 60000); 
});

/* ── Number formatting ────────────────────────────────── */
function fmt(n, decimals = 2) {
  if (n === null || n === undefined) return '—';
  return Number(n).toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals
  });
}

function fmtPct(n) {
  const v = Number(n);
  const sign = v >= 0 ? '+' : '';
  return `${sign}${v.toFixed(2)}%`;
}
