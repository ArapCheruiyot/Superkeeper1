import { auth, db } from "./firebase-config.js";
import { doc, getDoc, setDoc } from "https://www.gstatic.com/firebasejs/9.23.0/firebase-firestore.js";

// Get the navbar left div where username is
const navLeft = document.querySelector(".nav-left");

// Create an element to display shop name
const shopSpan = document.createElement("div");
shopSpan.id = "shop-name";
shopSpan.style.fontWeight = "500";
shopSpan.style.fontSize = "0.9rem";
shopSpan.style.marginTop = "4px";
navLeft.appendChild(shopSpan);

// Listen for auth state changes
auth.onAuthStateChanged(async (user) => {
    if (user) {
        const userDocRef = doc(db, "Shops", user.uid);
        const userSnapshot = await getDoc(userDocRef);

        if (userSnapshot.exists() && userSnapshot.data().shopName) {
            // Display existing shop name
            shopSpan.textContent = `Shop: ${userSnapshot.data().shopName}`;
        } else {
            // Prompt user to enter a shop name
            let shopName = "";
            while (!shopName) {
                shopName = prompt("Please enter your Shop name:");
                if (!shopName) alert("Shop name cannot be empty!");
            }
            // Save shop name in Firestore
            await setDoc(userDocRef, { shopName: shopName }, { merge: true });
            shopSpan.textContent = `Shop: ${shopName}`;
        }
    } else {
        // No user signed in, redirect to landing page
        window.location.href = "/";
    }
});
