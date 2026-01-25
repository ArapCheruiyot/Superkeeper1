// itemAdditionalMetadata.js
export function ensureItemMetadata(itemName, existing = {}) {
    const metadataSection = document.getElementById("item-metadata-section");
    if (!metadataSection) return;

    metadataSection.innerHTML = "";

    const wrapper = document.createElement("div");
    wrapper.style.display = "flex";
    wrapper.style.flexDirection = "column";
    wrapper.style.gap = "16px";

    function makePriceRow(label, value) {
        const row = document.createElement("div");
        row.className = "price-row";
        row.style.display = "flex";
        row.style.alignItems = "center";
        row.style.justifyContent = "space-between";
        row.style.padding = "8px 0";

        row.innerHTML = `
            <span class="price-label" style="font-weight:600;">${label}</span>

            <div style="display:flex; align-items:center; gap:8px;">
                <span class="price-value" style="font-weight:bold;">
                    ${value ? value : "Not set"}
                </span>

                <button class="price-edit-btn"
                        style="background:none; border:none; cursor:pointer;">
                    ✏️
                </button>
            </div>
        `;

        return row;
    }

    const buyingRow = makePriceRow("Buying Price", existing.buyingPrice);
    const sellingRow = makePriceRow("Selling Price", existing.sellingPrice);

    wrapper.appendChild(buyingRow);
    wrapper.appendChild(sellingRow);

    metadataSection.appendChild(wrapper);
}
