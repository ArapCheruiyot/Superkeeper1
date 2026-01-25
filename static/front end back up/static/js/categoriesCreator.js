// categoriesCreator.js
export function injectCategoryTools(container) {
    const wrapper = document.createElement("div");
    wrapper.style.display = "flex";
    wrapper.style.flexDirection = "row";
    wrapper.style.width = "100%";
    wrapper.style.height = "100%";
    wrapper.style.fontFamily = "inherit";

    const categoryPanel = document.createElement("div");
    categoryPanel.className = "w-64 min-w-64 h-full bg-white rounded-lg shadow p-4 flex flex-col";

    categoryPanel.innerHTML = `
        <h2 class="text-xl font-bold mb-4 text-gray-700">Categories</h2>
        <button id="create-category-btn"
            class="bg-blue-600 text-white w-full py-2 rounded-md shadow hover:bg-blue-700 mb-3 font-bold text-lg">
            + Add Category
        </button>
        <div id="category-list"
            class="mt-2 flex flex-col items-start flex-1 overflow-y-auto border p-3 rounded-md bg-gray-50">
            <p class="text-gray-400 text-left w-full">No categories yet.</p>
        </div>
    `;

    const rightSection = document.createElement("div");
    rightSection.className = "flex-1 p-4 text-gray-400 text-left";
    rightSection.innerHTML = `<p class="mt-10">Select a category or add items here later.</p>`;

    wrapper.appendChild(categoryPanel);
    wrapper.appendChild(rightSection);
    container.appendChild(wrapper);

    const list = categoryPanel.querySelector("#category-list");

    // Add main category
    categoryPanel.querySelector("#create-category-btn").addEventListener("click", () => {
        const name = prompt("Enter new category name:");
        if (!name) return;

        const newCat = createCategoryElement(name, 0);
        list.appendChild(newCat);

        const emptyMsg = list.querySelector("p");
        if (emptyMsg) emptyMsg.remove();
    });

    const levelColors = ["#1E3A8A", "#047857", "#B45309", "#7C3AED", "#6B7280"];

    function createCategoryElement(name, level) {
        const item = document.createElement("div");
        item.className = "category-block w-full p-2 rounded-md shadow mb-2 flex flex-col border relative";
        item.dataset.level = String(level);

        const row = document.createElement("div");
        row.className = "flex items-center gap-2";
        row.style.marginLeft = `${level * 20}px`;

        const expandBtn = document.createElement("button");
        expandBtn.textContent = "▶";
        expandBtn.style.fontSize = "12px";
        expandBtn.style.border = "none";
        expandBtn.style.background = "transparent";
        expandBtn.style.cursor = "pointer";
        expandBtn.title = "Collapse/Expand Subcategories";

        const addBtn = document.createElement("button");
        addBtn.type = "button";
        addBtn.textContent = "+";
        addBtn.className = "text-green-600 hover:text-green-700 font-bold text-lg";
        addBtn.title = "Add item or subcategory";

        const deleteBtn = document.createElement("button");
        deleteBtn.type = "button";
        deleteBtn.textContent = "×";
        deleteBtn.className = "text-red-500 hover:text-red-700 font-bold text-lg";
        deleteBtn.title = "Delete category and its children";

        const nameSpan = document.createElement("span");
        nameSpan.textContent = name;
        nameSpan.style.color = levelColors[level % levelColors.length];
        nameSpan.style.fontWeight = "900";
        nameSpan.style.fontSize = level === 0 ? "1.15rem" : "1.05rem";
        nameSpan.style.flex = "1";

        // Make category name itself clickable as an item
        nameSpan.classList.add("category-item");

        row.appendChild(expandBtn);
        row.appendChild(addBtn);
        row.appendChild(nameSpan);
        row.appendChild(deleteBtn);

        item.appendChild(row);

        const childrenContainer = document.createElement("div");
        childrenContainer.className = "children-container flex flex-col";
        item.appendChild(childrenContainer);

        expandBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            childrenContainer.hidden = !childrenContainer.hidden;
            expandBtn.textContent = childrenContainer.hidden ? "▶" : "▼";
        });

        addBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            openSubPanel(item, childrenContainer);
        });

        deleteBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            if (confirm(`Delete "${name}" and all its children?`)) {
                item.remove();
                if (list.children.length === 0) {
                    list.innerHTML = `<p class="text-gray-400 text-left w-full">No categories yet.</p>`;
                }
            }
        });

        item._addBtn = addBtn;
        item._deleteBtn = deleteBtn;
        item._nameSpan = nameSpan;
        item._expandBtn = expandBtn;
        item._childrenContainer = childrenContainer;

        return item;
    }

    function openSubPanel(categoryDiv, childrenContainer) {
        const existing = categoryDiv.querySelector(".sub-panel");
        if (existing) {
            setButtonsVisibility(categoryDiv, true);
            existing.remove();
            return;
        }

        setButtonsVisibility(categoryDiv, false);

        const panel = document.createElement("div");
        panel.className = "sub-panel mt-2 p-2 bg-gray-100 rounded flex flex-col gap-2";
        panel.style.zIndex = "10";
        panel.style.minWidth = "160px";

        const addItemBtn = document.createElement("button");
        addItemBtn.type = "button";
        addItemBtn.textContent = "Add Item";
        addItemBtn.className = "bg-blue-600 text-white px-2 py-1 rounded hover:bg-blue-700 font-bold";

        addItemBtn.addEventListener("click", () => {
            const itemName = prompt(`Enter item name for "${categoryDiv._nameSpan.textContent}":`);
            if (!itemName) return closePanel();

            const child = document.createElement("div");
            child.textContent = itemName;
            child.className = "font-bold p-1 category-item";

            childrenContainer.appendChild(child);
            closePanel();
        });

        const addSubBtn = document.createElement("button");
        addSubBtn.type = "button";
        addSubBtn.textContent = "Add Subcategory";
        addSubBtn.className = "bg-green-600 text-white px-2 py-1 rounded hover:bg-green-700 font-bold";

        addSubBtn.addEventListener("click", () => {
            const subName = prompt(`Enter subcategory name under "${categoryDiv._nameSpan.textContent}":`);
            if (!subName) return closePanel();

            const subCat = createCategoryElement(subName, Number(categoryDiv.dataset.level) + 1);
            childrenContainer.appendChild(subCat);
            closePanel();
        });

        function closePanel() {
            setButtonsVisibility(categoryDiv, true);
            panel.remove();
        }

        const onDocClick = (e) => {
            if (!panel.contains(e.target) && !categoryDiv.contains(e.target)) {
                closePanel();
                document.removeEventListener("click", onDocClick);
            }
        };

        setTimeout(() => document.addEventListener("click", onDocClick), 0);

        panel.appendChild(addItemBtn);
        panel.appendChild(addSubBtn);
        categoryDiv.appendChild(panel);
    }

    function setButtonsVisibility(categoryDiv, visible) {
        categoryDiv._addBtn.style.visibility = visible ? "" : "hidden";
        categoryDiv._deleteBtn.style.visibility = visible ? "" : "hidden";
        if (categoryDiv._expandBtn) categoryDiv._expandBtn.style.visibility = visible ? "" : "hidden";
    }

    return list;
}






import { loadPersistedCategories } from "./persistCategories.js";

// MAKE SURE these exist:
const shopId = "WHeQXgrqT0SEYJlrRx5OqZEUgYp2";
const categoryList = document.getElementById("category-list");

// Your existing function that creates <div class="category-block"> elements
import { createCategoryElement } from "./categoriesCreator.js";

// Call persistence loader
document.addEventListener("DOMContentLoaded", () => {
    console.log("Loading persisted categories...");
    loadPersistedCategories(shopId, categoryList, createCategoryElement);
});
