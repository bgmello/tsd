import os
import PyPDF2

def merge_pdfs(directory, output_filename):
    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    pdf_files.sort()

    pdf_writer = PyPDF2.PdfWriter()

    for pdf_file in pdf_files:
        print(pdf_file)
        pdf_path = os.path.join(directory, pdf_file)
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            pdf_writer.add_page(page)

    with open(output_filename, 'wb') as output_pdf:
        pdf_writer.write(output_pdf)

    print(f'Merged PDFs saved as {output_filename}')


if __name__ == '__main__':
    input_directory = 'data'
    output_file = 'results.pdf'
    merge_pdfs(input_directory, output_file)
