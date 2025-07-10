const loadScript = (src, callback)=>{
    const script = document.createElement("script");
    script.src = src;
    script.onload = callback;
    document.head.appendChild(script);
};
loadScript("https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js", () => 
console.log("htmltocanvasloaded"));
loadScript("https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js", () => 
console.log("pdftocanvasloaded"));
