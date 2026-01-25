// Import Firebase modules (v9 modular)
import { initializeApp } from "https://www.gstatic.com/firebasejs/9.23.0/firebase-app.js";
import { getAuth, GoogleAuthProvider } from "https://www.gstatic.com/firebasejs/9.23.0/firebase-auth.js";
import { getFirestore } from "https://www.gstatic.com/firebasejs/9.23.0/firebase-firestore.js";

// Your Firebase config
const firebaseConfig = {
  apiKey: "AIzaSyBO0xOY_O6uadK5pDhqlYHPcE2C-6cFlNo",
  authDomain: "superkeeper-ec181.firebaseapp.com",
  projectId: "superkeeper-ec181",
  storageBucket: "superkeeper-ec181.firebasestorage.app",
  messagingSenderId: "1070569435595",
  appId: "1:1070569435595:web:b63e591a8551ec802b064e",
  measurementId: "G-T4X41WYK95"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Initialize Services
const auth = getAuth(app);
const provider = new GoogleAuthProvider();
const db = getFirestore(app);

// Export modules for other scripts
export { app, auth, provider, db };
