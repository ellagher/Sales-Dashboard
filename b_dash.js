document.addEventListener("DOMContentLoaded", function (){console.log("load");
    if (typeof html2canvas === "undefined"){
        console.error("html2canvas is not loaded");
    } else {
        console.log("html2canvas is loaded", typeof html2canvas);
    }
    document.body.addEventListener("click", function (e){
        if (e.target && e.target.id === "export-image-btn"){
            html2canvas(document.getElementById("dashboard-image")).then(function (canvas){
                var link = document.createElement("a");
                link.download = "dashboard.png";
                link.href = canvas.toDataURL();
                link.click();
            });
        }
        if (e.target && e.target.id === "export-pdf-btn"){
            html2canvas(document.getElementById("dashboard-image")).then(function (canvas){
                const { jsPDF } = window.jspdf;
                const pdf = new jsPDF();
                const imgData = canvas.toDataURL("image/png");
                const imgProps = pdf.getImageProperties(imgData);
                const pdfWidth = pdf.internal.pageSize.getWidth();
                const pdfHeight = (imgProps.height*pdfWidth)/imgProps.width;
                pdf.addImage(imgData, "PNG", 0, 0, pdfWidth, pdfHeight);
                pdf.save("dashboard.pdf");
                
                
            });
        }
    })
});


