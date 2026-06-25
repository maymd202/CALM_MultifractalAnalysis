# CALM_MultifracalAnalysis

Code to replicate results in paper "Quantitative Validation of the “Coastline” Heuristic for Café-au-Lait Macule Borders in Neurofibromatosis Type 1 and McCune-Albright Syndrome"

## To fine-tune Segment Anything:
https://github.com/maymd202/micro-sam/CALM_sam_finetune.py
https://github.com/maymd202/micro-sam/CALM_predict_all.py

Code adapted from: Archit A, Freckmann L, Nair S, Khalid N, Hilt P, Rajashekar V, et al. Segment Anything for Microscopy. Nat Methods. 2025;22(3):579–91.

## Post-process CALMs:
https://github.com/maymd202/CALM_MultiFractalAnalysis/50_crop.py


## To create a fine-tuned classifier:
https://github.com/maymd202/peft/CALM_imclass.py

Code adapted from: Mangrulkar S, Gugger S, Debut L, Belkada Y, Paul S, Bossan B, et al. PEFT: State-of-the-art parameter efficient fine-tuning methods Github 2022 [https://github.com/huggingface/peft]

## Additional References

The following resources are relevant to the project's topic and may be useful for readers seeking additional background.

# Images Featured 

Jibbe N, Jibbe A, Rajpara A. McCune Albright Syndrome. Kans J Med. 2020;13:49–50.

Hernández-Martín A, Duat-Rodríguez A. An Update on Neurofibromatosis Type 1: Not Just Café-au-Lait 
Spots, Freckling, and Neurofibromas. An Update. Part I. Dermatological Clinical Criteria Diagnostic of the Disease. Actas Dermo-Sifiliográficas (English Edition). 2016;107(6):454–64.

Morata Alba J, Morata Alba L, Díez Gandía E. ¿Qué puede ocultar una mancha café con leche? Pediatría 
Atención Primaria. 2018;20:371–4.

Mao B, Chen S, Chen X, Yu X, Zhai X, Yang T, et al. Clinical characteristics and spectrum of NF1 
mutations in 12 unrelated Chinese families with neurofibromatosis type 1. BMC Medical Genetics. 2018;19(1):101.

Robinson C, Collins MT, Boyce AM. Fibrous Dysplasia/McCune-Albright Syndrome: Clinical and Translational Perspectives. Curr Osteoporos Rep. 2016;14(5):178–86.

Neri I, Lambertini M, Tengattini V, Rivalta B, Patrizi A. Halolike Phenomenon Around a Café au Lait Spot Superimposed on a Mongolian Spot. Pediatric Dermatology. 2017;34(3):e152–e3.

Agopiantz M, Journeau P, Lebon-Labich B, Sorlin A, Cuny T, Weryha G, et al. McCune–Albright syndrome, natural history and multidisciplinary management in a series of 14 pediatric cases. Annales d'Endocrinologie. 2016;77(1):7–13.

Supekar BB, Rambhia KD, Mukhi JI, Singh RP. Segmental giant café au lait macule in neurofibromatosis 1. Pigment International. 2018;5(1):50–3.

Bissonnette B, Luginbuehl I, Engelhardt T. McCune-Albright Syndrome. Syndromes: Rapid Recognition 
and Perioperative Implications, 2e. New York, NY: McGraw-Hill Education; 2019.

Moraes FS, Santos WEdM, Salomão GH. Neurofibromatosis type I. 2013;72(2):128–31.

Alonso DG. Young african woman with birthmark. Neurofibtomatosis. Shutterstock2018.

# Segmentation 

Chakkaravarthy AP, Chandrasekar A, editors. An Automatic Segmentation of Skin Lesion from 
Dermoscopy Images using Watershed Segmentation. 2018 International Conference on Recent Trends in Electrical, Control and Communication (RTECC); 2018 20–22 March 2018.

Phung SL, Bouzerdoum A, Chai D, editors. Skin segmentation using color and edge information. Seventh 
International Symposium on Signal Processing and Its Applications, 2003 Proceedings; 2003 4–4 July 2003.

Hosen MS, Zhang H. BAS-SegNet: A Boundary-Aware Sobel-Enhanced Deep Learning Framework for 
Breast Cancer and Skin Cancer Segmentation. Electronics. 2026;15(1):75.

Kirillov A, Mintun E, Ravi N, Mao H, Rolland C, Gustafson L, et al. Segment Anything. Proceedings of 
209 the IEEE/CVF international conference on computer vision (ICCV). 2023:4015–26.

Archit A, Freckmann L, Nair S, Khalid N, Hilt P, Rajashekar V, et al. Segment Anything for Microscopy. 
211 Nat Methods. 2025;22(3):579–91.

Yushkevich P, Piven J, Hazlett H, Smith R, Ho S, Gee J, et al. User-guided 3D active contour segmentation 
of anatomical structures significantly improved efficiency and reliability. Neuroimage. 2006;31(3):1116–28.

Taha AA, Hanbury A. Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool. 
BMC Med Imaging. 2015;15:29.

Lopes R, Betrouni N. Fractal and multifractal analysis: a review. Med Image Anal. 2009;13(4):634–49.

Karperien AL, Jelinek HF. Box-Counting Fractal Analysis: A Primer for the Clinician. Adv Neurobiol. 
2024;36:15–55.

# Fractal Analysis

Gould DJ, Vadakkan TJ, Poché RA, Dickinson ME. Multifractal and lacunarity analysis of microvascular 
220 morphology and remodeling. Microcirculation. 2011;18(2):136–51.

Long Y, Chen Y. Multifractal scaling analyses of urban street network structure: The cases of twelve 
megacities in China. PLOS ONE. 2021;16(2):e0246925.

# Classifier
Karperien A. FracLac for ImageJ. 2.5 ed. Bathurst, Australia: Charles Sturt University; 2013.
Mangrulkar S, Gugger S, Debut L, Belkada Y, Paul S, Bossan B, et al. PEFT: State-of-the-art parameter 
efficient fine-tuning methods Github2022 [https://github.com/huggingface/peft:[
Hu E, Shen Y, Wallis P, Allen-Zhu Z, Li Y, Wang S, et al. LoRA: Low-Rank Adaptation of Large 
Language Models. arXiv:210609685. 2021.
Yu W, Luo M, Zhou P, Si C, Zhou Y, Wang X, et al. MetaFormer Is Actually What You Need for Vision. 
arXiv:211111418. 2022.