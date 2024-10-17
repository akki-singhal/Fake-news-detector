function checkFakeNews() {
    var newsInput = document.getElementById('newsInput').value;
    var resultDiv = document.getElementById('result');

    // Simple check for demonstration purposes
    if (newsInput.includes("fake")) {
        resultDiv.style.color = 'red';
        resultDiv.innerText = 'Fake News Detected!';
    } else {
        resultDiv.style.color = 'green';
        resultDiv.innerText = 'News is likely genuine.';
    }
}
