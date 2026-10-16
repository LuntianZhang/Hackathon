const CACHE = 'pansense-shell-v13-phone-scan';
const SHELL = ['/', '/index.html', '/prototype.css', '/app.css', '/app.js', '/scan-desktop.js', '/scan-map.js', '/capture.css', '/icon.svg', '/icon-192.png', '/icon-512.png', '/manifest.webmanifest'];
self.addEventListener('install', event => {event.waitUntil(caches.open(CACHE).then(cache => cache.addAll(SHELL.map(url=>new Request(url,{cache:'reload'})))));self.skipWaiting();});
self.addEventListener('activate', event => {event.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(key => key !== CACHE).map(key => caches.delete(key)))));self.clients.claim();});
self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);
  // Never persist meal photos, personal records, nutrition data or 3D frames.
  if (url.origin !== self.location.origin || url.pathname.startsWith('/api/') || event.request.method !== 'GET') return;
  if (!SHELL.includes(url.pathname)) return;
  event.respondWith(fetch(event.request,{cache:'no-cache'}).then(response => {
    if (response.ok) {const copy=response.clone();caches.open(CACHE).then(cache => cache.put(event.request,copy));}
    return response;
  }).catch(() => caches.match(event.request,{ignoreSearch:true}).then(cached => cached || caches.match('/'))));
});
