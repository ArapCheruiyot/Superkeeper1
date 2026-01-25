import { injectCategoryTools } from "./categoriesCreator.js";

const manageStockBtn = document.getElementById("manage-stock-btn");

manageStockBtn.addEventListener("click", () => {
    const navbar = document.querySelector("nav.navbar");
    const navbarHeight = navbar.offsetHeight;

    const overlay = document.createElement("div");
    overlay.id = "manage-stock-overlay";

    overlay.style.position = "fixed";
    overlay.style.left = "0";
    overlay.style.top = navbarHeight + "px";
    overlay.style.width = "100%";
    overlay.style.height = `calc(100% - ${navbarHeight}px)`;
    overlay.style.backgroundColor = "white";
    overlay.style.zIndex = "5000";
    overlay.style.overflow = "auto";
    overlay.style.display = "flex";
    overlay.style.justifyContent = "flex-start";
    overlay.style.alignItems = "flex-start";

    overlay.innerHTML = `
        <div id="overlay-content" class="flex flex-col w-full h-full">
            <button id="close-overlay"
                class="absolute top-4 right-6 text-3xl font-bold text-gray-600 hover:text-red-500">
                &times;
            </button>

            <div class="p-6">
                <h2 class="text-2xl font-semibold mb-2 text-gray-700">Manage Your Stock</h2>
                <p class="text-gray-600 mb-4">Use the tools below to manage stock categories.</p>
            </div>

            <div id="dashboard-container" class="flex flex-row w-full h-full gap-4 p-6"></div>
        </div>
    `;

    document.body.appendChild(overlay);

    const dashboardContainer = overlay.querySelector("#dashboard-container");
    injectCategoryTools(dashboardContainer);

    overlay.querySelector("#close-overlay").addEventListener("click", () => {
        overlay.remove();
    });
});
