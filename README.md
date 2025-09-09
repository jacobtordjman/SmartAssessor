# SmartAssessor

## Run Backend on Google Colab, Frontend Locally

1) Open `SmartAssessor_Colab.ipynb` in Google Colab and run the cells in order:
   - Installs Python deps and optionally mounts Google Drive.
   - Starts FastAPI on port 8000 with CORS set for local dev (`http://localhost:5173,http://127.0.0.1:5173`).
   - Launches a Cloudflare Tunnel and prints a public URL like `https://<subdomain>.trycloudflare.com`.

2) Start the frontend locally:
   - `cd frontend`
   - `npm install`
   - `npm run dev` → open `http://localhost:5173`

3) Point the frontend to the Colab backend URL (pick one):
   - Browser console at `http://localhost:5173`:
     - `localStorage.setItem('API_BASE', 'https://<your-public-url>')` then refresh
   - Or edit `frontend/public/config.js` and set `window.API_BASE` to the public URL
   - Or create `frontend/.env.local` with `VITE_API_BASE=https://<your-public-url>`

Notes
- The backend only accepts PDF uploads at `POST /upload/assessment`.
- The model is large; the code automatically falls back to a lightweight grader if it cannot load the 7B weights.
- Colab sessions are ephemeral; keep the tunnel cell running to keep the URL alive.
