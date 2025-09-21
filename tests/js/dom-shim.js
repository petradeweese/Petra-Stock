const { setTimeout, clearTimeout, setInterval, clearInterval } = global;

class ClassList {
  constructor(element) {
    this.element = element;
    this._classes = new Set();
  }

  _sync() {
    if (this.element) {
      const value = Array.from(this._classes).join(' ');
      if (value) {
        this.element.attributes.class = value;
      } else {
        delete this.element.attributes.class;
      }
    }
  }

  add(...tokens) {
    tokens.forEach((token) => {
      if (!token) {
        return;
      }
      const value = String(token).trim();
      if (!value) {
        return;
      }
      this._classes.add(value);
    });
    this._sync();
  }

  remove(...tokens) {
    tokens.forEach((token) => {
      if (!token) {
        return;
      }
      this._classes.delete(String(token).trim());
    });
    this._sync();
  }

  contains(token) {
    if (!token) {
      return false;
    }
    return this._classes.has(String(token).trim());
  }

  toggle(token, force) {
    const value = String(token).trim();
    if (!value) {
      return false;
    }
    if (force === true) {
      this._classes.add(value);
      this._sync();
      return true;
    }
    if (force === false) {
      this._classes.delete(value);
      this._sync();
      return false;
    }
    if (this._classes.has(value)) {
      this._classes.delete(value);
      this._sync();
      return false;
    }
    this._classes.add(value);
    this._sync();
    return true;
  }

  toString() {
    return Array.from(this._classes).join(' ');
  }

  set(value) {
    this._classes.clear();
    if (value) {
      String(value)
        .split(/\s+/)
        .map((part) => part.trim())
        .filter(Boolean)
        .forEach((part) => this._classes.add(part));
    }
    this._sync();
  }
}

class NodeBase {
  constructor(ownerDocument) {
    this.ownerDocument = ownerDocument;
    this.childNodes = [];
    this.parentNode = null;
    this._textContent = '';
  }

  appendChild(node) {
    if (!(node instanceof NodeBase)) {
      return null;
    }
    if (node.parentNode) {
      node.parentNode.removeChild(node);
    }
    this.childNodes.push(node);
    node.parentNode = this;
    return node;
  }

  removeChild(node) {
    const idx = this.childNodes.indexOf(node);
    if (idx !== -1) {
      this.childNodes.splice(idx, 1);
      node.parentNode = null;
    }
    return node;
  }

  insertBefore(node, referenceNode) {
    if (!(node instanceof NodeBase)) {
      return null;
    }
    if (node.parentNode) {
      node.parentNode.removeChild(node);
    }
    const idx = this.childNodes.indexOf(referenceNode);
    if (idx === -1) {
      return this.appendChild(node);
    }
    this.childNodes.splice(idx, 0, node);
    node.parentNode = this;
    return node;
  }

  get firstChild() {
    return this.childNodes[0] || null;
  }

  get lastChild() {
    return this.childNodes[this.childNodes.length - 1] || null;
  }
}

class ElementNode extends NodeBase {
  constructor(tagName, ownerDocument) {
    super(ownerDocument);
    this.tagName = tagName.toUpperCase();
    this.attributes = {};
    this.classList = new ClassList(this);
    this.dataset = {};
    this.style = {};
    this.eventListeners = {};
    this.hidden = false;
    this.disabled = false;
    this.scrollTop = 0;
    this.clientHeight = 0;
    this.clientWidth = 0;
  }

  setAttribute(name, value) {
    const attr = String(value);
    this.attributes[name] = attr;
    if (name === 'id') {
      this.ownerDocument._registerId(attr, this);
    }
    if (name === 'class') {
      this.classList.set(attr);
    }
    if (name === 'hidden') {
      this.hidden = attr === 'true' || attr === '';
    }
  }

  getAttribute(name) {
    return Object.prototype.hasOwnProperty.call(this.attributes, name)
      ? this.attributes[name]
      : null;
  }

  append(...nodes) {
    nodes.forEach((node) => {
      if (node instanceof NodeBase) {
        this.appendChild(node);
      } else if (node !== undefined && node !== null) {
        this._textContent += String(node);
      }
    });
  }

  get textContent() {
    const childText = this.childNodes
      .map((child) => (child instanceof ElementNode ? child.textContent : ''))
      .join('');
    return `${this._textContent || ''}${childText}`;
  }

  set textContent(value) {
    this._textContent = value ? String(value) : '';
    this.childNodes = [];
  }

  get innerHTML() {
    return this.textContent;
  }

  set innerHTML(value) {
    this.textContent = value;
  }

  get parentElement() {
    return this.parentNode instanceof ElementNode ? this.parentNode : null;
  }

  get children() {
    return this.childNodes.filter((child) => child instanceof ElementNode);
  }

  get className() {
    return this.classList.toString();
  }

  set className(value) {
    this.classList.set(value);
  }

  get id() {
    return this.getAttribute('id') || '';
  }

  set id(value) {
    this.setAttribute('id', value);
  }

  addEventListener(type, handler) {
    if (!this.eventListeners[type]) {
      this.eventListeners[type] = [];
    }
    this.eventListeners[type].push(handler);
  }

  removeEventListener(type, handler) {
    const listeners = this.eventListeners[type];
    if (!listeners) {
      return;
    }
    const idx = listeners.indexOf(handler);
    if (idx !== -1) {
      listeners.splice(idx, 1);
    }
  }

  dispatchEvent(event) {
    if (!event || !event.type) {
      return;
    }
    const listeners = this.eventListeners[event.type] || [];
    listeners.forEach((listener) => {
      try {
        listener.call(this, event);
      } catch (err) {
        // ignore listener errors in test shim
      }
    });
  }

  getBoundingClientRect() {
    const height = this._rectHeight || this.clientHeight || 0;
    const width = this._rectWidth || this.clientWidth || 0;
    return {
      top: 0,
      left: 0,
      right: width,
      bottom: height,
      width,
      height,
    };
  }

  querySelectorAll(selector) {
    const results = [];
    const matcher = createMatcher(selector);
    const traverse = (node) => {
      if (!(node instanceof ElementNode)) {
        return;
      }
      if (matcher(node)) {
        results.push(node);
      }
      node.childNodes.forEach((child) => traverse(child));
    };
    this.childNodes.forEach((child) => traverse(child));
    return results;
  }

  querySelector(selector) {
    const matches = this.querySelectorAll(selector);
    return matches[0] || null;
  }
}

class ButtonElement extends ElementNode {}

class CanvasElement extends ElementNode {
  constructor(ownerDocument) {
    super('canvas', ownerDocument);
    this.width = 0;
    this.height = 0;
  }

  getContext(type) {
    if (type !== '2d') {
      return null;
    }
    const noop = () => {};
    return {
      save: noop,
      restore: noop,
      scale: noop,
      clearRect: noop,
      beginPath: noop,
      moveTo: noop,
      lineTo: noop,
      stroke: noop,
      fill: noop,
      setLineDash: noop,
      arc: noop,
      strokeStyle: '#000',
      fillStyle: '#000',
      lineWidth: 1,
    };
  }
}

function createMatcher(selector) {
  if (!selector) {
    return () => false;
  }
  if (selector.startsWith('.')) {
    const cls = selector.slice(1);
    return (node) => node.classList.contains(cls);
  }
  if (selector.startsWith('#')) {
    const id = selector.slice(1);
    return (node) => node.id === id;
  }
  const tag = selector.toUpperCase();
  return (node) => node.tagName === tag;
}

class DocumentNode extends ElementNode {
  constructor() {
    super('#document', null);
    this._ids = new Map();
    this.eventListeners = {};
    this.documentElement = new ElementNode('html', this);
    this.body = new ElementNode('body', this);
    this.documentElement.appendChild(this.body);
    this.visibilityState = 'visible';
    this.appendChild(this.documentElement);
  }

  createElement(tagName) {
    const tag = String(tagName).toLowerCase();
    if (tag === 'button') {
      return new ButtonElement('button', this);
    }
    if (tag === 'canvas') {
      return new CanvasElement(this);
    }
    return new ElementNode(tag, this);
  }

  getElementById(id) {
    return this._ids.get(id) || null;
  }

  _registerId(id, element) {
    if (!id) {
      return;
    }
    this._ids.set(id, element);
  }

  addEventListener(type, handler) {
    if (!this.eventListeners[type]) {
      this.eventListeners[type] = [];
    }
    this.eventListeners[type].push(handler);
  }

  removeEventListener(type, handler) {
    const listeners = this.eventListeners[type];
    if (!listeners) {
      return;
    }
    const idx = listeners.indexOf(handler);
    if (idx !== -1) {
      listeners.splice(idx, 1);
    }
  }
}

function buildBaseDom(document) {
  const card = document.createElement('div');
  card.className = 'card forward-card';
  document.body.appendChild(card);

  const loading = document.createElement('div');
  loading.id = 'forward-loading';
  card.appendChild(loading);

  const empty = document.createElement('div');
  empty.id = 'forward-empty';
  empty.hidden = true;
  card.appendChild(empty);

  const error = document.createElement('div');
  error.id = 'forward-error';
  error.hidden = true;
  card.appendChild(error);

  const table = document.createElement('table');
  table.id = 'forward-table';
  table.hidden = true;
  card.appendChild(table);

  const tbody = document.createElement('tbody');
  tbody.id = 'forward-tbody';
  table.appendChild(tbody);

  const runBtn = document.createElement('button');
  runBtn.id = 'forward-run';
  runBtn.disabled = true;
  card.appendChild(runBtn);

  const toast = document.createElement('div');
  toast.id = 'toast';
  toast.hidden = true;
  document.body.appendChild(toast);

  const historyModal = document.createElement('div');
  historyModal.id = 'forward-history-modal';
  historyModal.hidden = true;
  document.body.appendChild(historyModal);

  const dialog = document.createElement('div');
  dialog.className = 'forward-history-dialog';
  historyModal.appendChild(dialog);

  const header = document.createElement('div');
  header.className = 'forward-history-header';
  dialog.appendChild(header);

  const title = document.createElement('h2');
  title.id = 'forward-history-title';
  header.appendChild(title);

  const closeBtn = document.createElement('button');
  closeBtn.className = 'forward-history-close';
  header.appendChild(closeBtn);

  const toolbar = document.createElement('div');
  toolbar.className = 'forward-history-toolbar';
  dialog.appendChild(toolbar);

  const exportLink = document.createElement('a');
  exportLink.id = 'forward-history-export';
  toolbar.appendChild(exportLink);

  const body = document.createElement('div');
  body.className = 'forward-history-body';
  dialog.appendChild(body);

  const status = document.createElement('div');
  status.id = 'forward-history-status';
  status.hidden = true;
  body.appendChild(status);

  const scroll = document.createElement('div');
  scroll.className = 'forward-history-scroll';
  scroll.clientHeight = 240;
  body.appendChild(scroll);

  const list = document.createElement('ul');
  list.id = 'forward-history-list';
  scroll.appendChild(list);

  const footer = document.createElement('div');
  footer.className = 'forward-history-footer';
  dialog.appendChild(footer);

  const loadMore = document.createElement('button');
  loadMore.id = 'forward-history-load-more';
  footer.appendChild(loadMore);
}

function makeFavorite(id) {
  return {
    id,
    ticker: 'AAPL',
    direction: 'UP',
    interval: '15m',
    lookback_display: '1.0y',
    target_pct: 1.0,
    stop_pct: 0.5,
    window_value: 4,
    window_unit: 'Hours',
    forward_history: [],
    summary: {
      n: 0,
      hits: 0,
      hit_rate: 0,
      hit_lb95: 0,
      avg_roi: 0,
      median_tt_bars: 0,
      avg_dd: 0,
      mode: 'off',
      half_life_days: 30,
      unweighted: {
        n: 0,
        hits: 0,
        hit_rate: 0,
        hit_lb95: 0,
        avg_roi: 0,
        median_tt_bars: 0,
        avg_dd: 0,
      },
      weighted: {
        n: 0,
        hits: 0,
        hit_rate: 0,
        hit_lb95: 0,
        avg_roi: 0,
        median_tt_bars: 0,
        avg_dd: 0,
      },
    },
    forward: null,
  };
}

function createTestEnv() {
  const document = new DocumentNode();
  buildBaseDom(document);

  const window = {
    document,
    navigator: { userAgent: 'node' },
    console,
    setTimeout,
    clearTimeout,
    setInterval,
    clearInterval,
    requestAnimationFrame: (fn) => setTimeout(() => fn(Date.now()), 16),
    cancelAnimationFrame: (id) => clearTimeout(id),
    devicePixelRatio: 1,
    getComputedStyle: () => ({
      getPropertyValue: () => '',
    }),
    addEventListener(type, handler) {
      if (!this._listeners) {
        this._listeners = {};
      }
      if (!this._listeners[type]) {
        this._listeners[type] = [];
      }
      this._listeners[type].push(handler);
    },
    removeEventListener(type, handler) {
      if (!this._listeners || !this._listeners[type]) {
        return;
      }
      const listeners = this._listeners[type];
      const idx = listeners.indexOf(handler);
      if (idx !== -1) {
        listeners.splice(idx, 1);
      }
    },
    __FORWARD_TEST_MODE__: true,
  };

  document.defaultView = window;
  window.window = window;
  window.document = document;
  window.HTMLElement = ElementNode;
  window.HTMLButtonElement = ButtonElement;
  window.Node = NodeBase;
  window.Element = ElementNode;
  window.URLSearchParams = URLSearchParams;
  window.location = { href: 'http://localhost/' };

  return {
    window,
    document,
    console,
    setTimeout,
    clearTimeout,
    setInterval,
    clearInterval,
    HTMLElement: ElementNode,
    HTMLButtonElement: ButtonElement,
    Node: NodeBase,
    globalThis: window,
    URLSearchParams,
    makeFavorite,
  };
}

module.exports = { createTestEnv, makeFavorite };
