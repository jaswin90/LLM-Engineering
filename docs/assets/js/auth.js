// Google sign-in (Firebase Auth) + progress sync (Cloud Firestore).
// Loaded as an ES module after book.js. If firebase-config.js is empty the book
// keeps working with browser-local progress only.
import { firebaseConfig } from './firebase-config.js';

const FB = 'https://www.gstatic.com/firebasejs/10.12.2/';
const BookAuth = { configured: false, user: null, signIn() {}, signOut() {} };
window.BookAuth = BookAuth;

function whenBookReady() {
  return new Promise(res => { if (window.Book && document.querySelector('.topbar')) res(); else document.addEventListener('book:ready', () => res(), { once: true }); });
}

async function main() {
  await whenBookReady();
  const Book = window.Book;
  if (!firebaseConfig || !firebaseConfig.apiKey) { Book.renderAuth(null); rebindLocal(Book); return; }

  const [{ initializeApp }, authMod, fsMod] = await Promise.all([
    import(FB + 'firebase-app.js'), import(FB + 'firebase-auth.js'), import(FB + 'firebase-firestore.js')
  ]);
  const { getAuth, GoogleAuthProvider, signInWithPopup, signInWithRedirect, onAuthStateChanged, signOut } = authMod;
  const { getFirestore, doc, getDoc, setDoc, serverTimestamp } = fsMod;

  const app = initializeApp(firebaseConfig);
  const auth = getAuth(app);
  const db = getFirestore(app);
  const provider = new GoogleAuthProvider();
  BookAuth.configured = true;

  BookAuth.signIn = async () => {
    try { await signInWithPopup(auth, provider); }
    catch (e) {
      if (e && (e.code === 'auth/popup-blocked' || e.code === 'auth/operation-not-supported-in-this-environment')) await signInWithRedirect(auth, provider);
      else { console.error(e); Book.toast('Sign-in failed: ' + (e.code || e.message)); }
    }
  };
  BookAuth.signOut = async () => { await signOut(auth); Book.toast('Signed out — progress stays in this browser.'); };

  let pending = {}; let timer = null; let ready = false;
  function queue(ev) {
    if (!BookAuth.user || !ready) return;
    if (ev.kind === 'reload') { pushAll(); return; }
    const bucket = ev.kind === 'chapter' ? 'chapters' : 'items';
    pending[bucket] = pending[bucket] || {};
    pending[bucket][ev.id] = ev.rec === null ? fsMod.deleteField() : ev.rec;
    clearTimeout(timer); timer = setTimeout(flush, 800);
  }
  async function flush() {
    const u = BookAuth.user; if (!u) return;
    const patch = pending; pending = {};
    if (!Object.keys(patch).length) return;
    try { await setDoc(doc(db, 'users', u.uid), Object.assign(patch, { updatedAt: serverTimestamp() }), { merge: true }); }
    catch (e) { console.error('sync failed', e); Book.toast('Could not sync progress (offline?) — will retry on next change.'); }
  }
  async function pushAll() {
    const u = BookAuth.user; if (!u) return;
    const d = Book.store.data;
    try { await setDoc(doc(db, 'users', u.uid), { items: d.items, chapters: d.chapters, updatedAt: serverTimestamp(), profile: { name: u.displayName || '', email: u.email || '' } }); }
    catch (e) { console.error(e); }
  }
  Book.store.onChange(queue);

  onAuthStateChanged(auth, async user => {
    BookAuth.user = user; ready = false;
    Book.renderAuth(user, false);
    if (!user) return;
    try {
      const snap = await getDoc(doc(db, 'users', user.uid));
      if (snap.exists()) Book.store.merge(snap.data());
      ready = true;
      await pushAll();
      Book.renderAuth(user, true);
      const note = document.getElementById('sync-note'); if (note) note.textContent = 'Synced to your Google account';
      Book.toast('Progress synced with your Google account');
    } catch (e) { console.error(e); Book.toast('Signed in, but sync failed: ' + (e.code || e.message)); }
  });
}

function rebindLocal(Book) {
  const b = document.getElementById('auth-btn');
  if (b) b.addEventListener('click', () => Book.toast('Google sign-in is not configured on this deployment yet — progress is saved in this browser.'));
}

main().catch(e => console.error(e));
