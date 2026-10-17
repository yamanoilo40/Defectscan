// Кэш для офлайн-работы.
// Код приложения и labels.json: сначала сеть (обновления приходят сразу), без сети — из кэша.
// Модель (адрес с ?v=версия) и библиотека ONNX: из кэша, новая версия модели = новый адрес = скачивается заново.
const V = 'defectscan-v3';
const SHELL = ['./', 'index.html', 'style.css', 'app.js', 'manifest.webmanifest', 'icons/icon-192.png', 'icons/icon-512.png'];
const NET_FIRST = /(\/|index\.html|app\.js|style\.css|labels\.json|manifest\.webmanifest)$/;
self.addEventListener('install', e => { e.waitUntil(caches.open(V).then(c => Promise.allSettled(SHELL.map(u => c.add(u)))).then(() => self.skipWaiting())); });
self.addEventListener('activate', e => { e.waitUntil(caches.keys().then(k => Promise.all(k.filter(x => x !== V).map(x => caches.delete(x)))).then(() => self.clients.claim())); });
self.addEventListener('fetch', e => {
  if (e.request.method !== 'GET') return;
  const u = new URL(e.request.url);
  const own = u.origin === location.origin;
  if (!own && !u.host.endsWith('jsdelivr.net')) return;
  const isModel = own && /model\.onnx$/.test(u.pathname);
  if (isModel && !u.searchParams.has('v')) return;                      // без версии модель не кэшируем
  if (own && NET_FIRST.test(u.pathname)) {
    e.respondWith(fetch(e.request).then(r => { if (r.ok) { const c = r.clone(); caches.open(V).then(x => x.put(e.request, c)); } return r; })
                                  .catch(() => caches.match(e.request)));
    return;
  }
  e.respondWith(caches.open(V).then(async c => {
    const hit = await c.match(e.request); if (hit) return hit;
    const r = await fetch(e.request); if (r && (r.ok || r.type === 'opaque')) c.put(e.request, r.clone()); return r;
  }));
});
