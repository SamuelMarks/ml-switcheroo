/** 
 * op_tabs.js
 * 
 * Handles switching logic for the Vertical Tabs documentation UI. 
 * Designed to support multiple tab containers on a single page if needed, 
 * though typically operation pages are singular. 
 */ 

function openOpTab(evt, tabName) { 
    // 1. Traverse up to the container to scope queries 
    // This allows multiple tab groups on one page safely 
    var container = evt.currentTarget.closest(".op-tabs-container"); 
    if (!container) return; 

    var i, tabcontent, tablinks; 

    // 2. Hide all tab panes in THIS container 
    tabcontent = container.getElementsByClassName("op-tab-pane"); 
    for (i = 0; i < tabcontent.length; i++) { 
        tabcontent[i].style.display = "none"; 
        tabcontent[i].classList.remove("active"); 
    } 

    // 3. Deactivate all buttons in THIS container 
    tablinks = container.getElementsByClassName("op-tab-btn"); 
    for (i = 0; i < tablinks.length; i++) { 
        tablinks[i].classList.remove("active"); 
    } 

    // 4. Activate current selection 
    var targetPane = container.querySelector(`[id="${tabName}"]`); 
    if (targetPane) { 
        targetPane.style.display = "block"; 
        targetPane.classList.add("active"); 
    } 
    evt.currentTarget.classList.add("active"); 
}