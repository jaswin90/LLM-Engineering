# Setup: publishing the book and enabling Google sign-in

## 1. Publish on GitHub Pages (one-time, ~1 minute)

1. Open **https://github.com/jaswin90/LLM-Engineering/settings/pages**
2. Under **Build and deployment → Source**, choose **GitHub Actions**.
3. Open the **Actions** tab, select **Deploy book to GitHub Pages**, and click **Run workflow** (or push any change under `docs/`).
4. The book is then live at **https://jaswin90.github.io/LLM-Engineering/**

The workflow (`.github/workflows/pages.yml`) uploads the `docs/` folder as-is. There is no build step: every page is plain HTML/CSS/JS.

> Alternative: choose **Deploy from a branch**, pick `claude/funny-hopper-lcw8ww` (or `main` after merging) and the `/docs` folder. `docs/.nojekyll` is already present.

## 2. Where readers' progress lives

- **Default (no setup):** answers, ratings and completed chapters are saved in the reader's own browser (localStorage). Nothing is sent anywhere.
- **Sync link / file (no setup, already live):** on *My progress*, "Copy sync link" packs the reader's progress into the link's `#` fragment (deflate-compressed, about 6 KB even with every question answered). Opening the link on another device merges it in, keeping the newer answer per question. Browsers never send the `#` part to the server, so no server stores anything. "Export file" / "Import file" do the same with a JSON file.
- **Google sign-in (optional, below):** automatic sync across devices, stored in your Firebase project.

## 3. Enable "Sign in with Google" and cloud-synced progress (optional, ~10 minutes)

Without this step the book works fully, with browser storage and sync links. With it, signed-in readers get their history on every device automatically.

1. Go to https://console.firebase.google.com and **create a project** (Analytics not needed).
2. **Build → Authentication → Get started → Sign-in method → Google → Enable.** Set a support email and save.
3. **Authentication → Settings → Authorized domains → Add domain:** `jaswin90.github.io`
4. **Build → Firestore Database → Create database** (production mode; pick a region near your readers, e.g. `asia-south1` Mumbai).
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

## 4. Preview locally

```bash
cd docs && python3 -m http.server 8000
```

Then open http://localhost:8000. (ES modules do not load from `file://`, so use a local server.)
