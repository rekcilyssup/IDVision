/* ==========================================
   IDVision Premium Dashboard JS Controller
   ========================================== */

document.addEventListener("DOMContentLoaded", () => {
    // State management
    let currentCardImageBase64 = null;
    let stats = {
        clean_count: 0,
        degraded_count: 0,
        has_results: false,
        has_chart: false
    };

    // DOM Elements - Navigation Stats
    const cleanCountEl = document.getElementById("cleanCount");
    const degradedCountEl = document.getElementById("degradedCount");
    const evalStatusEl = document.getElementById("evalStatus");

    // DOM Elements - Card Generator
    const form = document.getElementById("generatorForm");
    const cardNameInput = document.getElementById("cardName");
    const cardIdInput = document.getElementById("cardId");
    const cardDobInput = document.getElementById("cardDob");
    const cardAddressInput = document.getElementById("cardAddress");
    const cardIssueInput = document.getElementById("cardIssue");
    const cardExpiryInput = document.getElementById("cardExpiry");
    const cardLayoutSelect = document.getElementById("cardLayout");
    const cardThemeSelect = document.getElementById("cardTheme");

    const btnRandomize = document.getElementById("btnRandomize");
    const btnGenerate = document.getElementById("btnGenerate");
    const btnSaveToClean = document.getElementById("btnSaveToClean");

    const cardImagePlaceholder = document.getElementById("cardImagePlaceholder");
    const renderedCardImg = document.getElementById("renderedCardImg");
    const renderedCardJson = document.getElementById("renderedCardJson");

    // Viewer Tabs
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabPanels = document.querySelectorAll(".tab-panel");

    // DOM Elements - Degradation Playground
    const degTypeRadios = document.getElementsByName("degType");
    const btnApplyDegradation = document.getElementById("btnApplyDegradation");
    const playgroundSourceImg = document.getElementById("playgroundSourceImg");
    const playgroundSourcePlaceholder = document.getElementById("playgroundSourcePlaceholder");
    const playgroundDegradedImg = document.getElementById("playgroundDegradedImg");
    const playgroundDegradedPlaceholder = document.getElementById("playgroundDegradedPlaceholder");

    // Sliders & Visibility Groups
    const sliderGroupBlur = document.getElementById("sliderGroupBlur");
    const sliderGroupJpeg = document.getElementById("sliderGroupJpeg");
    const sliderGroupRotation = document.getElementById("sliderGroupRotation");

    const blurRadiusInput = document.getElementById("blurRadius");
    const jpegQualityInput = document.getElementById("jpegQuality");
    const rotationAngleInput = document.getElementById("rotationAngle");

    const blurValEl = document.getElementById("blurVal");
    const jpegValEl = document.getElementById("jpegVal");
    const rotationValEl = document.getElementById("rotationVal");

    // DOM Elements - Evaluation
    const btnRunEval = document.getElementById("btnRunEval");
    const consoleLog = document.getElementById("consoleLog");
    const consoleStatus = document.getElementById("consoleStatus");
    const metricsChart = document.getElementById("metricsChart");
    const metricsChartPlaceholder = document.getElementById("metricsChartPlaceholder");

    // Initialize Page
    fetchStats();
    randomizeFields();

    // ==========================================
    // Fetch statistics and results state
    // ==========================================
    async function fetchStats() {
        try {
            const res = await fetch("/api/stats");
            const data = await res.json();
            stats = data;
            
            cleanCountEl.innerText = data.clean_count;
            degradedCountEl.innerText = data.degraded_count;
            
            if (data.has_results) {
                evalStatusEl.innerText = "Evaluated";
                evalStatusEl.classList.add("evaluated");
            } else {
                evalStatusEl.innerText = "Not Evaluated";
                evalStatusEl.classList.remove("evaluated");
            }

            if (data.has_chart) {
                metricsChart.src = `/results/comparison_chart.png?t=${Date.now()}`;
                metricsChart.classList.remove("hidden");
                metricsChartPlaceholder.classList.add("hidden");
            } else {
                metricsChart.classList.add("hidden");
                metricsChartPlaceholder.classList.remove("hidden");
            }
        } catch (err) {
            console.error("Error fetching statistics:", err);
        }
    }

    // ==========================================
    // Tab Toggling
    // ==========================================
    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            tabBtns.forEach(b => b.classList.remove("active"));
            tabPanels.forEach(p => p.classList.remove("active"));

            btn.classList.add("active");
            const targetId = btn.getAttribute("data-target");
            document.getElementById(targetId).classList.add("active");
        });
    });

    // ==========================================
    // Randomize Card Fields
    // ==========================================
    function randomizeFields() {
        const firstNames = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Elizabeth", "William", "Linda", "David", "Barbara", "Richard", "Susan"];
        const lastNames = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson"];
        const streets = ["Main St", "Oak Ave", "Pine Rd", "Maple Dr", "Cedar Ln", "Elm St", "View Rd", "Park Meadow Dr", "Sunset Blvd"];
        const cities = ["San Francisco", "Austin", "Seattle", "Chicago", "Boston", "Denver", "Miami", "New York", "Los Angeles"];
        const states = ["CA", "TX", "WA", "IL", "MA", "CO", "FL", "NY", "CA"];
        
        const randomElement = arr => arr[Math.floor(Math.random() * arr.length)];
        const randomRange = (min, max) => Math.floor(Math.random() * (max - min + 1)) + min;
        const padZero = num => num.toString().padStart(2, "0");

        // Set random name
        const name = `${randomElement(firstNames)} ${randomElement(lastNames)}`.toUpperCase();
        cardNameInput.value = name;

        // Set random ID Number
        const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        const idNum = `${randomElement(chars)}${randomElement(chars)}${randomRange(10000, 99999)}${randomElement(chars)}${randomElement(chars)}`;
        cardIdInput.value = `DL-${idNum}`;

        // Random birth date (between 1970 and 2003)
        const dobYear = randomRange(1970, 2003);
        const dobMonth = padZero(randomRange(1, 12));
        const dobDay = padZero(randomRange(1, 28));
        cardDobInput.value = `${dobYear}-${dobMonth}-${dobDay}`;

        // Random address
        cardAddressInput.value = `${randomRange(100, 9999)} ${randomElement(streets).toUpperCase()}, ${randomElement(cities).toUpperCase()}, ${randomElement(states)} ${randomRange(10000, 99999)}`;

        // Random issue date and expiry date
        const issueYear = randomRange(2020, 2025);
        const issueMonth = padZero(randomRange(1, 12));
        const issueDay = padZero(randomRange(1, 28));
        cardIssueInput.value = `${issueYear}-${issueMonth}-${issueDay}`;

        const expiryYear = issueYear + randomRange(5, 10);
        cardExpiryInput.value = `${expiryYear}-${issueMonth}-${issueDay}`;

        // Random layout and theme selection
        cardLayoutSelect.value = randomRange(0, 2);
        cardThemeSelect.value = randomRange(0, 4);
    }

    btnRandomize.addEventListener("click", randomizeFields);

    // ==========================================
    // Generate Card API Request
    // ==========================================
    async function generateCard(saveToClean = false) {
        // Prepare request body
        const payload = {
            name: cardNameInput.value,
            id_number: cardIdInput.value,
            dob: cardDobInput.value,
            address: cardAddressInput.value,
            issue_date: cardIssueInput.value,
            expiry_date: cardExpiryInput.value,
            layout: cardLayoutSelect.value,
            theme: cardThemeSelect.value,
            save_to_clean: saveToClean
        };

        btnGenerate.disabled = true;
        btnSaveToClean.disabled = true;
        if (saveToClean) {
            btnSaveToClean.innerText = "Saving...";
        } else {
            btnGenerate.innerText = "Generating...";
        }

        try {
            const res = await fetch("/api/generate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            
            // Set output state
            currentCardImageBase64 = data.image_b64;

            // Render output image
            renderedCardImg.src = data.image_b64;
            renderedCardImg.classList.remove("hidden");
            cardImagePlaceholder.classList.add("hidden");

            // Render JSON schema
            renderedCardJson.innerText = JSON.stringify(data.record, null, 2);

            // Populate Playground Clean Source image
            playgroundSourceImg.src = data.image_b64;
            playgroundSourceImg.classList.remove("hidden");
            playgroundSourcePlaceholder.classList.add("hidden");

            // Hide previous degraded image
            playgroundDegradedImg.classList.add("hidden");
            playgroundDegradedPlaceholder.classList.remove("hidden");

            if (saveToClean) {
                fetchStats();
            }
        } catch (err) {
            console.error("Card generation failed:", err);
            alert("Card generation failed. Check backend logs.");
        } finally {
            btnGenerate.disabled = false;
            btnSaveToClean.disabled = false;
            btnGenerate.innerText = "Generate Card";
            btnSaveToClean.innerText = "Save to Clean Set";
        }
    }

    btnGenerate.addEventListener("click", () => generateCard(false));
    btnSaveToClean.addEventListener("click", () => generateCard(true));

    // ==========================================
    // Degradation Playground Sliders displays
    // ==========================================
    function updateSliderDisplays() {
        blurValEl.innerText = parseFloat(blurRadiusInput.value).toFixed(1);
        jpegValEl.innerText = parseInt(jpegQualityInput.value);
        rotationValEl.innerText = parseFloat(rotationAngleInput.value).toFixed(1) + "°";
    }

    blurRadiusInput.addEventListener("input", updateSliderDisplays);
    jpegQualityInput.addEventListener("input", updateSliderDisplays);
    rotationAngleInput.addEventListener("input", updateSliderDisplays);

    // Toggle slider controls depending on selected corruption type
    degTypeRadios.forEach(radio => {
        radio.addEventListener("change", (e) => {
            // Update labels styles
            document.querySelectorAll(".radio-btn").forEach(btn => btn.classList.remove("active"));
            e.target.closest(".radio-btn").classList.add("active");

            const filterType = e.target.value;
            
            // Hide all sliders first
            sliderGroupBlur.classList.add("hidden");
            sliderGroupJpeg.classList.add("hidden");
            sliderGroupRotation.classList.add("hidden");

            if (filterType === "blur") {
                sliderGroupBlur.classList.remove("hidden");
            } else if (filterType === "jpeg") {
                sliderGroupJpeg.classList.remove("hidden");
            } else if (filterType === "rotation") {
                sliderGroupRotation.classList.remove("hidden");
            }
        });
    });

    // ==========================================
    // Apply Dynamic Degradation Image filter API
    // ==========================================
    btnApplyDegradation.addEventListener("click", async () => {
        if (!currentCardImageBase64) {
            alert("Please generate a clean ID document card first.");
            return;
        }

        const selectedType = document.querySelector('input[name="degType"]:checked').value;
        const payload = {
            image: currentCardImageBase64,
            type: selectedType,
            blur_radius: blurRadiusInput.value,
            jpeg_quality: jpegQualityInput.value,
            rotation_angle: rotationAngleInput.value
        };

        btnApplyDegradation.innerText = "Applying Filter...";
        btnApplyDegradation.disabled = true;

        try {
            const res = await fetch("/api/degrade", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const data = await res.json();

            playgroundDegradedImg.src = data.image_b64;
            playgroundDegradedImg.classList.remove("hidden");
            playgroundDegradedPlaceholder.classList.add("hidden");
        } catch (err) {
            console.error("Filter corruption failed:", err);
            alert("Image corruption simulation failed.");
        } finally {
            btnApplyDegradation.innerText = "Simulate Corruption";
            btnApplyDegradation.disabled = false;
        }
    });

    // ==========================================
    // Run Evaluation Benchmark Process
    // ==========================================
    btnRunEval.addEventListener("click", async () => {
        btnRunEval.disabled = true;
        btnRunEval.innerText = "Running Harness...";
        consoleStatus.innerText = "Running";
        consoleStatus.classList.add("running");
        consoleLog.innerText = "Triggering Qwen2-VL Evaluation Harness script (Dry Run mode)...\n";

        try {
            const res = await fetch("/api/run-eval", {
                method: "POST"
            });
            const data = await res.json();

            if (data.status === "success") {
                consoleLog.innerText += `\nEvaluation finished successfully!\n\n${data.log}`;
                consoleStatus.innerText = "Idle";
                consoleStatus.classList.remove("running");
                consoleStatus.classList.add("completed");
                
                // Refresh counts and chart
                fetchStats();
            } else {
                consoleLog.innerText += `\nError occurred:\n${data.error}`;
                consoleStatus.innerText = "Failed";
                consoleStatus.classList.remove("running");
            }
        } catch (err) {
            consoleLog.innerText += `\nNetwork execution failure: ${err.message}`;
            consoleStatus.innerText = "Failed";
            consoleStatus.classList.remove("running");
        } finally {
            btnRunEval.disabled = false;
            btnRunEval.innerText = "Run Pipeline Evaluation";
        }
    });
});
