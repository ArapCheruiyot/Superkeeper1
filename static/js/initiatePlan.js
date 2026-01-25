// static/js/initiatePlan.js
import { getAuth, onAuthStateChanged } from "https://www.gstatic.com/firebasejs/9.23.0/firebase-auth.js";

document.addEventListener("DOMContentLoaded", () => {
  const auth = getAuth();

  onAuthStateChanged(auth, async (user) => {
    if (user) {
      console.log("[INITIATE PLAN] User logged in:", user.uid);

      // Get current active shop from localStorage (your existing logic)
      const shopId = localStorage.getItem("activeShopId");
      if (!shopId) {
        console.warn("[INITIATE PLAN] No activeShopId found in localStorage");
        return;
      }

      try {
        const response = await fetch("/ensure-plan", {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({ shop_id: shopId })
        });

        const data = await response.json();

        if (data.created) {
          console.log(`[INITIATE PLAN] Default plan initialized for shop ${shopId}`);
        } else {
          console.log(`[INITIATE PLAN] Plan already exists for shop ${shopId}`);
        }
      } catch (err) {
        console.error("[INITIATE PLAN] Error ensuring plan:", err);
      }
    }
  });
});
