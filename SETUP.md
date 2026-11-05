# Setup: publishing the book and enabling Google sign-in

## 1. Publish on GitHub Pages (one-time, ~1 minute)

1. Open **https://github.com/jaswin90/LLM-Engineering/settings/pages**
2. Under **Build and deployment → Source**, choose **GitHub Actions**.
3. Open the **Actions** tab, select **Deploy book to GitHub Pages**, and click **Run workflow** (or push any change under `docs/`).
4. The book is then live at **https://jaswin90.github.io/LLM-Engineering/**

The workflow (`.github/workflows/pages.yml`) uploads the `docs/` folder as-is. There is no build step: every page is plain HTML/CSS/JS.

> Alternative: choose **Deploy from a branch**, pick `claude/funny-hopper-lcw8ww` (or `main` after merging) and the `/docs` folder. `docs/.nojekyll` is already present.

## 2. Enable "Sign in with Google" and cloud-synced progress (optional, ~10 minutes)

Without this step, readers' answers are saved in their own browser (localStorage). With it, signed-in readers get their history on every device.

1. Go to https://console.firebase.google.com and **create a project** (Analytics not needed).
2. **Build → Authentication → Get started → Sign-in method → Google → Enable.** Set a support email and save.
3. **Authentication → Settings → Authorized domains → Add domain:** `jaswin90.github.io`
4. **Build → Firestore Database → Create database** (production mode, any region).
5. **Firestore → Rules**, paste and publish:

   ```
   rules_version = '2';
   service cloud.firestore {
     match /databases/{database}/documents {
       match /users/{uid} {
         allow read, write: if request.auth != null && request.auth.uid == uid;
       }
     }
   }
   ```

6. **Project settings (gear) → Your apps → Web (`</>`)** → register an app → copy the `firebaseConfig` object.
7. Paste the values into `docs/assets/js/firebase-config.js` and push. (These values identify the project; they are not secrets. Access is controlled by the rules in step 5.)

Each reader's progress is a single Firestore document `users/{uid}` containing their MCQ answers, interview-question ratings and chapter completion. Free-tier Firestore limits are far above what a book needs.

## 3. Preview locally

```bash
cd docs && python3 -m http.server 8000
```

Then open http://localhost:8000. (ES modules do not load from `file://`, so use a local server.)
