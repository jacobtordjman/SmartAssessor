// src/firebase.ts
import { initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";

export const firebaseConfig = {
  apiKey: "AIzaSyDACu87dFEck5OnN56lT3qIDaIl-5ptQvg",
  authDomain: "smart-assessor-fdeaa.firebaseapp.com",
  projectId: "smart-assessor-fdeaa",
  storageBucket: "smart-assessor-fdeaa.firebasestorage.app",
  messagingSenderId: "764336991828",
  appId: "1:764336991828:web:24d9804f0643d8833d47c1",
  measurementId: "G-DCW0PSJW0Q",
};

export const app = initializeApp(firebaseConfig);
export const db = getFirestore(app);
