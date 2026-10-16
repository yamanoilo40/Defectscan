// Кэш для офлайн-работы: оболочка приложения, модель и библиотека ONNX Runtime.
const V='defectscan-v1';
const SHELL=['./','index.html','style.css','app.js','manifest.webmanifest','model/labels.json','model/model.onnx','icons/icon-192.png','icons/icon-512.png'];
self.addEventListener('install',e=>{e.waitUntil(caches.open(V).then(c=>Promise.allSettled(SHELL.map(u=>c.add(u)))).then(()=>self.skipWaiting()))});
self.addEventListener('activate',e=>{e.waitUntil(caches.keys().then(k=>Promise.all(k.filter(x=>x!==V).map(x=>caches.delete(x)))).then(()=>self.clients.claim()))});
self.addEventListener('fetch',e=>{
  const u=new URL(e.request.url); if(e.request.method!=='GET') return;
  if(u.origin!==location.origin && !u.host.endsWith('jsdelivr.net')) return;
  e.respondWith(caches.open(V).then(async c=>{
    const hit=await c.match(e.request);
    const net=fetch(e.request).then(r=>{ if(r&&(r.ok||r.type==='opaque')) c.put(e.request,r.clone()); return r }).catch(()=>hit);
    return hit||net;
  }));
});
