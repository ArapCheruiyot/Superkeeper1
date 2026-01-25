// itemDetails/itemNameEditor.js
export function attachItemNameEditor() {
    const overlayTitle = document.getElementById("overlay-item-name");
    if (!overlayTitle) return;

    // Avoid adding multiple pencils
    if (overlayTitle.querySelector(".edit-icon")) return;

    const pencil = document.createElement("span");
    pencil.textContent = "✏️"; // small pencil emoji
    pencil.className = "edit-icon ml-2 cursor-pointer text-lg";
    pencil.title = "Edit item name";

    pencil.addEventListener("click", () => {
        const currentName = overlayTitle.textContent.trim();
        const input = document.createElement("input");
        input.type = "text";
        input.value = currentName;
        input.className = "border px-2 py-1 rounded text-lg";
        overlayTitle.textContent = "";
        overlayTitle.appendChild(input);
        input.focus();

        const finishEdit = () => {
            overlayTitle.textContent = input.value.trim() || currentName;
            overlayTitle.appendChild(pencil);
        };

        input.addEventListener("blur", finishEdit);
        input.addEventListener("keydown", (e) => {
            if (e.key === "Enter") input.blur();
            if (e.key === "Escape") {
                overlayTitle.textContent = currentName;
                overlayTitle.appendChild(pencil);
            }
        });
    });

    overlayTitle.appendChild(pencil);
}
