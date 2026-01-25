// itemImageCapture.js
// Ensures overlay has image UI and captures images per item (prevents bleed across items)

export function ensureItemImages(itemName = "") {
    const overlayContent = document.getElementById("item-overlay-content");
    if (!overlayContent) return null;

    // small normalizer to map "British Bread" -> "britishbread" (stable key)
    function normalizeKey(str) {
        return String(str || "")
            .replace(/[^a-zA-Z0-9]/g, "")
            .trim()
            .toLowerCase();
    }

    const itemKey = normalizeKey(itemName);

    // Create a flex wrapper for images + details if missing
    let flexWrapper = overlayContent.querySelector(".overlay-flex-wrapper");
    if (!flexWrapper) {
        flexWrapper = document.createElement("div");
        flexWrapper.classList.add("overlay-flex-wrapper");
        flexWrapper.style.display = "flex";
        flexWrapper.style.flexDirection = "row"; // images left, details right
        flexWrapper.style.gap = "20px"; // spacing between images and details
        overlayContent.appendChild(flexWrapper);
    }

    // Create image container if missing
    let container = flexWrapper.querySelector(".overlay-image-container");
    if (!container) {
        container = document.createElement("div");
        container.classList.add("overlay-image-container");
        container.style.display = "flex";
        container.style.flexDirection = "column"; // vertical stack
        container.style.gap = "10px";
        container.style.width = "120px"; // small width for left column
        flexWrapper.appendChild(container);
    }

    // If container is for a different item, clear images (but keep capture button)
    const existingKey = container.dataset.itemKey || "";
    if (existingKey !== itemKey) {
        // remove any <img> elements but keep buttons (like capture button)
        Array.from(container.querySelectorAll("img")).forEach(img => img.remove());
        // mark container as belonging to current item
        container.dataset.itemKey = itemKey;
    }

    // Create camera trigger button if missing
    let cameraButton = container.querySelector(".capture-camera-btn");
    if (!cameraButton) {
        cameraButton = document.createElement("button");
        cameraButton.textContent = "Capture";
        cameraButton.classList.add("capture-camera-btn");
        cameraButton.style.padding = "6px 10px";
        cameraButton.style.borderRadius = "5px";
        cameraButton.style.border = "none";
        cameraButton.style.backgroundColor = "#3b82f6";
        cameraButton.style.color = "white";
        cameraButton.style.cursor = "pointer";
        container.appendChild(cameraButton);
    }

    // Helper to get current images (array of data URLs / src)
    function getImages() {
        return Array.from(container.querySelectorAll("img")).map(i => i.src);
    }

    // Handler: show file picker and append images (keeps them on container.dataset.itemKey)
    cameraButton.onclick = () => {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = "image/*";
        input.multiple = true;
        input.capture = "environment";

        input.addEventListener("change", (event) => {
            const files = Array.from(event.target.files || []);
            files.forEach(file => {
                const reader = new FileReader();
                reader.onload = (e) => {
                    const img = document.createElement("img");
                    img.src = e.target.result;
                    img.alt = "Captured Item Image";
                    img.style.width = "100%";  // fills container width (small)
                    img.style.height = "auto"; // maintain aspect ratio
                    img.style.objectFit = "cover";
                    img.style.border = "1px solid #ccc";
                    img.style.borderRadius = "4px";

                    // append before the button (so button stays at bottom) — optional
                    container.insertBefore(img, cameraButton);
                };
                reader.readAsDataURL(file);
            });

            // reset input so the same file(s) can be picked again if needed
            event.target.value = "";
        });

        input.click();
    };

    // Expose an object to the caller for convenience
    return {
        container,
        getImages,
        itemKey
    };
}
