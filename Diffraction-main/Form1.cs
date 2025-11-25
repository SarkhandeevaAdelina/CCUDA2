using System;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;
using System.Drawing;
using System.Text;
using DifrOnLenta = Diffraction.Program.DifrOnLenta;
using Compl = Diffraction.Program.Compl;

namespace Diffraction
{
    public partial class MainForm : Form
    {
        public MainForm()
        {
            InitializeComponent();

            // Настройка графика chartRealPart
            chartRealPart.Series[0].ChartType = SeriesChartType.Spline;
            chartRealPart.Series[0].Color = Color.Blue;
            chartRealPart.Series[0].BorderWidth = 3;
            chartRealPart.Series[0].Name = "Полное поле u без скин-слоя";
            chartRealPart.Series.Add(new Series());

            chartRealPart.Series[1].ChartType = SeriesChartType.Spline;
            chartRealPart.Series[1].Color = Color.FromArgb(180, 255, 0, 0); // Полупрозрачный красный
            chartRealPart.Series[1].BorderWidth = 3;
            chartRealPart.Series[1].BorderDashStyle = ChartDashStyle.Solid; // Сплошная линия
            chartRealPart.Series[1].Name = "Полное поле u со скин-слоем";

            // Включение и настройка легенды
            chartRealPart.Legends[0].Enabled = true;
            chartRealPart.Legends[0].Docking = Docking.Bottom;
            chartRealPart.Legends[0].Alignment = StringAlignment.Center;

            // Подписи осей
            chartRealPart.ChartAreas[0].AxisX.Title = "x";
            chartRealPart.ChartAreas[0].AxisY.Title = "y";

            // Настройка внешнего вида осей
            chartRealPart.ChartAreas[0].AxisX.TitleFont = new Font("Arial", 10, FontStyle.Bold);
            chartRealPart.ChartAreas[0].AxisY.TitleFont = new Font("Arial", 10, FontStyle.Bold);
            
            // Автоматический запуск формы с графиками при загрузке
            this.Shown += MainForm_Shown;
        }

        // Обработчик события Shown для автоматического открытия Form2
        private void MainForm_Shown(object sender, EventArgs e)
        {
            // Автоматически открываем форму с графиками
            OpenGraphicsForm();
        }

        // Метод для открытия формы с графиками (рефакторинг button2_Click)
        private void OpenGraphicsForm()
        {
            // Создание нового экземпляра формы Form2
            Form2 form2 = new Form2();

            // Создание изображений графиков с помощью метода CreateGraphImages
            GraphImagePair images = CreateGraphImages(form2);

            if (images != null)
            {
                form2.pictureBoxNoSkin.Image = images.ImageNoSkin;
                form2.pictureBoxSkin.Image = images.ImageSkin;
            }

            form2.Show(); // Используем Show() вместо ShowDialog() для неблокирующего отображения
        }

        // Обработчик события изменения значения числового поля xL.
        private void numericUpDown1_ValueChanged(object sender, EventArgs e)
        {
            // Если значение числового поля xL больше или равно значению числового поля xR,
            // установить значение числового поля xL равным значению числового поля xR - 1.
            if (xL.Value >= xR.Value)
            {
                xL.Value = xR.Value - 1;
            }
            // Установить минимальное значение оси X графика chartRealPart равным значению числового поля xL.
            chartRealPart.ChartAreas[0].AxisX.Minimum = (int)xL.Value;
        }

        // Обработчик события изменения значения числового поля yDn.
        private void numericUpDown2_ValueChanged(object sender, EventArgs e)
        {
            // Если значение числового поля yDn больше или равно значению числового поля yUp,
            // установить значение числового поля yDn равным значению числового поля yUp - 1.
            if (yDn.Value >= yUp.Value)
            {
                yDn.Value = yUp.Value - 1;
            }
            // Установить минимальное значение оси Y графика chartRealPart равным значению числового поля yDn.
            chartRealPart.ChartAreas[0].AxisY.Minimum = (int)yDn.Value;
        }

        // Обработчик события изменения значения числового поля xR.
        private void numericUpDown3_ValueChanged(object sender, EventArgs e)
        {
            // Если значение числового поля xL больше или равно значению числового поля xR,
            // установить значение числового поля xR равным значению числового поля xL + 1.
            if (xL.Value >= xR.Value)
            {
                xR.Value = xL.Value + 1;
            }
            // Установить максимальное значение оси X графика chartRealPart равным значению числового поля xR.
            chartRealPart.ChartAreas[0].AxisX.Maximum = (int)xR.Value;
        }

        // Обработчик события изменения значения числового поля yUp.
        private void numericUpDown4_ValueChanged(object sender, EventArgs e)
        {
            // Если значение числового поля yDn больше или равно значению числового поля yUp,
            // установить значение числового поля yUp равным значению числового поля yDn + 1.
            if (yDn.Value >= yUp.Value)
            {
                yUp.Value = yDn.Value + 1;
            }
            // Установить максимальное значение оси Y графика chartRealPart равным значению числового поля yUp.
            chartRealPart.ChartAreas[0].AxisY.Maximum = (int)yUp.Value;
        }

        // Обработчик события изменения значения числового поля truncationParameterN.
        private void numericUpDown8_ValueChanged(object sender, EventArgs e)
        {
            // Если значение числового поля truncationParameterN меньше или равно нулю,
            // установить значение числового поля truncationParameterN равным 1.
            if (truncationParameterN.Value <= 0)
            {
                truncationParameterN.Value = 1;
            }
        }

        // Обработчик события изменения значения числового поля bandBoundaryA.
        private void numericUpDown5_ValueChanged(object sender, EventArgs e)
        {
            // Если значение числового поля bandBoundaryA больше или равно значению числового поля bandBoundaryB,
            // установить значение числового поля bandBoundaryA равным значению числового поля bandBoundaryB - 1.
            if (bandBoundaryA.Value >= bandBoundaryB.Value)
            {
                bandBoundaryA.Value = bandBoundaryB.Value - 1;
            }
        }

        // Обработчик события изменения значения числового поля bandBoundaryB.
        private void numericUpDown6_ValueChanged(object sender, EventArgs e)
        {
            // Если значение числового поля bandBoundaryA больше или равно значению числового поля bandBoundaryB,
            // установить значение числового поля bandBoundaryB равным значению числового поля bandBoundaryA + 1.
            if (bandBoundaryA.Value >= bandBoundaryB.Value)
            {
                bandBoundaryB.Value = bandBoundaryA.Value + 1;
            }
        }

        // Обработчик события изменения значения числового поля wavelength.
        private void numericUpDown7_ValueChanged(object sender, EventArgs e)
        {
            // Если значение числового поля wavelength меньше или равно нулю,
            // установить значение числового поля wavelength равным 1.
            if (wavelength.Value <= 0)
            {
                wavelength.Value = 1;
            }
        }

        // Обработчик события нажатия кнопки button1.
        private void button1_Click(object sender, EventArgs e)
        {
            // Очистка предыдущих данных
            chartRealPart.Series[0].Points.Clear(); // Без скин-слоя
            chartRealPart.Series[1].Points.Clear(); // Со скин-слоем
            textBoxChebPolynomial.Clear();

            // Параметры задачи
            int param = (int)truncationParameterN.Value;
            double a = (double)bandBoundaryA.Value;
            double b = (double)bandBoundaryB.Value;
            double angle = (double)angleInDegrees.Value / 180 * Math.PI;
            double len = (double)wavelength.Value;
            double skinDepth = (double)skinDepthInput.Value;

            // Решение БЕЗ скин-слоя (skinDepth = 0)
            DifrOnLenta qNoSkin = new DifrOnLenta(a, b, len, angle, param, 0);
            if (qNoSkin.SolveDifr() != 0)
            {
                // Построение графика для случая без скин-слоя
                double h = (b - a) / 400, x = a + h;
                while (x < (b - h))
                {
                    chartRealPart.Series[0].Points.AddXY(x, qNoSkin.u(x, 0).Re);
                    x += h;
                }
            }

            // Решение С УЧЕТОМ скин-слоя
            DifrOnLenta qSkin = new DifrOnLenta(a, b, len, angle, param, skinDepth);
            if (qSkin.SolveDifr() != 0)
            { 
                // Вывод в текстовое поле коэффициентов разложения по полиномам Чебышева
                // ДЛЯ ОБОИХ СЛУЧАЕВ (без скина и со скином)
                textBoxChebPolynomial.Clear();
                
                // Заголовок с информацией о коэффициенте χ
                textBoxChebPolynomial.Text += "Импедансный коэффициент χ:" + Environment.NewLine;
                textBoxChebPolynomial.Text += string.Format("χ = {0:F6} + {1:F6}i{2}", qSkin.chi.Re, qSkin.chi.Im, Environment.NewLine);
                textBoxChebPolynomial.Text += Environment.NewLine;
                
                // Вывод ВСЕХ коэффициентов Чебышева (до param = N)
                int totalCoefficients = param;  // Получаем значение N из параметра усечения
                textBoxChebPolynomial.Text += string.Format("Коэффициенты Чебышева (всего {0}):{1}", totalCoefficients, Environment.NewLine);
                textBoxChebPolynomial.Text += string.Format("{0,-4} {1,-25} {2,-25}{3}", "#", "БЕЗ скин-слоя", "СО скин-слоем", Environment.NewLine);
                textBoxChebPolynomial.Text += new string('.', 60) + Environment.NewLine;
                
                // Выводим ВСЕ коэффициенты от 0 до param-1
                for (int coeffIndex = 0; coeffIndex < totalCoefficients; coeffIndex++)
                {
                    string noSkinCoeff = string.Format("{0:F4}+{1:F4}i", qNoSkin.y[coeffIndex].Re, qNoSkin.y[coeffIndex].Im);
                    string skinCoeff = string.Format("{0:F4}+{1:F4}i", qSkin.y[coeffIndex].Re, qSkin.y[coeffIndex].Im);
                    textBoxChebPolynomial.Text += string.Format("{0,-4} {1,-25} {2,-25}{3}", 
                        coeffIndex + 1, noSkinCoeff, skinCoeff, Environment.NewLine);
                }

                // Построение графика для случая со скин-слоем
                double h = (b - a) / 400, x = a + h;
                while (x < (b - h))
                {
                    chartRealPart.Series[1].Points.AddXY(x, qSkin.u(x, 0).Re);
                    x += h;
                }


                // Расчет проводимости материала
                try
                {
                    double skinDepth_m = skinDepth * 1e-3; // Переводим мм в метры
                    double wavelength_m = len;
                    double conductivity = qSkin.CalculateConductivity(skinDepth_m, wavelength_m);

                    lblConductivity.Text = string.Format("Проводимость: {0:E2} См/м", conductivity);
                    lblConductivity.ForeColor = Color.DarkBlue;

                    // Выводим сообщение с результатом
                    MessageBox.Show(
                        string.Format("Рассчитанная проводимость материала: {0:E2} См/м\n", conductivity) +
                        "Типичные значения для металлов: 10^6 - 10^7 См/м",
                        "Результат расчета проводимости",
                        MessageBoxButtons.OK,
                        MessageBoxIcon.Information
                    );
                }
                catch (Exception ex)
                {
                    MessageBox.Show(
                        string.Format("Ошибка расчета проводимости: {0}", ex.Message),
                        "Ошибка",
                        MessageBoxButtons.OK,
                        MessageBoxIcon.Error
                    );
                    lblConductivity.Text = "Ошибка расчета проводимости";
                    lblConductivity.ForeColor = Color.Red;
                }

                // Расчет энергий и проверка закона сохранения энергии
                var energyComp = qSkin.CalculateEnergyComponents();
                
                double incidentEnergy = energyComp.Incident;
                double reflectedEnergy = energyComp.Reflected;
                double transmittedEnergy = energyComp.Transmitted;
                double absorbedEnergy = energyComp.Absorbed;
                
                // Проверка физичности результатов
                double sumRAT = reflectedEnergy + absorbedEnergy + transmittedEnergy;
                
                // Относительные доли энергии
                double reflectedFraction = reflectedEnergy / incidentEnergy;
                double transmittedFraction = transmittedEnergy / incidentEnergy;
                double absorbedFraction = absorbedEnergy / incidentEnergy;

                // Формирование сообщения для всплывающего окна
                StringBuilder energyMessage = new StringBuilder();
                energyMessage.AppendLine("Проверка закона сохранения энергии:");
                energyMessage.AppendLine();
                
                if (energyComp.WasRenormalized)
                {
                    energyMessage.AppendLine("⚠ Выполнена ренормализация энергий (численные погрешности)");
                    energyMessage.AppendLine();
                }
                
                energyMessage.AppendLine(string.Format("Падающая энергия:     {0:F6} (100.00%)", incidentEnergy));
                energyMessage.AppendLine(string.Format("Отраженная энергия:   {0:F6} ({1:P2})", reflectedEnergy, reflectedFraction));
                energyMessage.AppendLine(string.Format("Прошедшая энергия:    {0:F6} ({1:P2})", transmittedEnergy, transmittedFraction));
                energyMessage.AppendLine(string.Format("Поглощенная энергия:  {0:F6} ({1:P2})", absorbedEnergy, absorbedFraction));
                energyMessage.AppendLine();
                energyMessage.AppendLine(string.Format("Сумма (отр+прош+погл): {0:F6}", sumRAT));
                energyMessage.AppendLine();
                
                // Проверка энергетического баланса
                double balanceError = Math.Abs(incidentEnergy - sumRAT);
                double relativeError = balanceError / Math.Max(incidentEnergy, 1e-10);
                
                bool energyConservationOk = relativeError < 0.05; // погрешность менее 5%
                
                if (energyConservationOk)
                {
                    energyMessage.AppendLine("✓ Энергетический баланс соблюден!");
                    energyMessage.AppendLine(string.Format("  Падающая энергия ≈ Сумма (отр+прош+погл)"));
                    energyMessage.AppendLine(string.Format("  Погрешность: {0:P2} (абс: {1:F6})", relativeError, balanceError));
                }
                else
                {
                    energyMessage.AppendLine("⚠ Внимание: заметное расхождение баланса");
                    energyMessage.AppendLine(string.Format("  Погрешность: {0:P2} (абс: {1:F6})", relativeError, balanceError));
                    energyMessage.AppendLine("  Возможные причины: численные погрешности интегрирования");
                }
                
                energyMessage.AppendLine();
                energyMessage.AppendLine("Физические проверки:");
                
                // Проверка физичности каждой компоненты
                bool allPositive = (reflectedEnergy >= 0) && (transmittedEnergy >= 0) && (absorbedEnergy >= 0);
                bool allLessOrEqual = (reflectedEnergy <= incidentEnergy) && 
                                     (transmittedEnergy <= incidentEnergy) && 
                                     (absorbedEnergy <= incidentEnergy);
                
                if (allPositive && allLessOrEqual)
                {
                    energyMessage.AppendLine("✓ Все компоненты энергии физичны");
                }
                else
                {
                    if (!allPositive)
                        energyMessage.AppendLine("✗ Обнаружены отрицательные значения энергии!");
                    if (!allLessOrEqual)
                        energyMessage.AppendLine("✗ Компонента энергии превышает падающую!");
                }

                // Показываем всплывающее окно с результатами проверки
                MessageBoxIcon icon = energyConservationOk ? MessageBoxIcon.Information : MessageBoxIcon.Warning;
                MessageBox.Show(
                    energyMessage.ToString(),
                    "Проверка закона сохранения энергии",
                    MessageBoxButtons.OK,
                    icon
                );
            }
            else
            {
                MessageBox.Show(
                    "Ошибка решения задачи с учетом скин-слоя!",
                    "Ошибка",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
            }

        }




        // Обработчик события нажатия кнопки button2
        private void button2_Click(object sender, EventArgs e)
        {
            OpenGraphicsForm();
        }


        // Метод для создания изображения графика.
        private class GraphImagePair
        {
            public Bitmap ImageNoSkin;
            public Bitmap ImageSkin;
        }

        private GraphImagePair CreateGraphImages(Form2 form2)
        {
            // Создание экземпляра класса DifrOnLenta с параметрами, заданными значениями числовых полей
            double y1 = (double)yDn.Value, y2 = (double)yUp.Value;
            double x1 = (double)xL.Value, x2 = (double)xR.Value;
            int param = (int)truncationParameterN.Value;
            double a = (double)bandBoundaryA.Value;
            double b = (double)bandBoundaryB.Value;
            double angle = (double)angleInDegrees.Value / 180 * Math.PI;
            double len = (double)wavelength.Value;
            double skinDepth = (double)skinDepthInput.Value;

            // Решение без скин-слоя
            DifrOnLenta qNoSkin = new DifrOnLenta(a, b, len, angle, param, 0);
            if (qNoSkin.SolveDifr() == 0)
            {
                return null;
            }

            // Решение со скин-слоем
            DifrOnLenta qSkin = new DifrOnLenta(a, b, len, angle, param, skinDepth);
            if (qSkin.SolveDifr() == 0)
            {
                return null;
            }

            // Создание массива значений функции u(x, y) на плоскости (x, y).
            int w = form2.pictureBoxNoSkin.Width;
            int h = form2.pictureBoxNoSkin.Height;

            double[,] uNoSkin = new double[w, h];
            double[,] uSkin = new double[w, h];

            double uMaxNoSkin = 0;
            double uMaxSkin = 0;

            for (int i = 0; i < w; i++)
            {
                double x = x1 + i / (double)w * (x2 - x1);
                for (int j = 0; j < h; j++)
                {
                    double y = y1 + j / (double)h * (y2 - y1);
                    uNoSkin[i, j] = Compl.Abs(qNoSkin.u(x, y));
                    uSkin[i, j] = Compl.Abs(qSkin.u(x, y));
                    if (uMaxNoSkin < uNoSkin[i, j])
                        uMaxNoSkin = uNoSkin[i, j];
                    if (uMaxSkin < uSkin[i, j])
                        uMaxSkin = uSkin[i, j];
                }
            }

            // Создание изображений графиков.
            Bitmap imageNoSkin = new Bitmap(w, h);
            Bitmap imageSkin = new Bitmap(w, h);

            for (int i = 1; i < w; i++)
            {
                for (int j = 1; j < h; j++)
                {
                    int colNoSkin = (int)(uNoSkin[i, j] / uMaxNoSkin * 255);
                    int colSkin = (int)(uSkin[i, j] / uMaxSkin * 255);
                    Color colorNoSkin = Color.FromArgb(colNoSkin, colNoSkin, colNoSkin);
                    Color colorSkin = Color.FromArgb(colSkin, colSkin, colSkin);

                    // Рисуем пиксели для поля без скин-слоя и со скин-слоем
                    imageNoSkin.SetPixel(i, j, colorNoSkin);
                    imageSkin.SetPixel(i, j, colorSkin);
                }
            }

            // Добавление на изображение графика точек, соответствующих границам ленты.
            int xC_A = (int)((a - x1) / (x2 - x1) * w);
            int xC_B = (int)((b - x1) / (x2 - x1) * w);
            int yC = h / 2;

            // Рисуем метки для границы A (левый край ленты)
            SafeSetPixel(imageNoSkin, xC_A - 2, yC - 2, Color.Black);
            SafeSetPixel(imageNoSkin, xC_A - 2, yC + 2, Color.Black);
            SafeSetPixel(imageNoSkin, xC_A + 2, yC - 2, Color.Black);
            SafeSetPixel(imageNoSkin, xC_A + 2, yC + 2, Color.Black);

            SafeSetPixel(imageSkin, xC_A - 2, yC - 2, Color.Black);
            SafeSetPixel(imageSkin, xC_A - 2, yC + 2, Color.Black);
            SafeSetPixel(imageSkin, xC_A + 2, yC - 2, Color.Black);
            SafeSetPixel(imageSkin, xC_A + 2, yC + 2, Color.Black);

            // Рисуем метки для границы B (правый край ленты)
            SafeSetPixel(imageNoSkin, xC_B - 2, yC - 2, Color.Black);
            SafeSetPixel(imageNoSkin, xC_B - 2, yC + 2, Color.Black);
            SafeSetPixel(imageNoSkin, xC_B + 2, yC - 2, Color.Black);
            SafeSetPixel(imageNoSkin, xC_B + 2, yC + 2, Color.Black);

            SafeSetPixel(imageSkin, xC_B - 2, yC - 2, Color.Black);
            SafeSetPixel(imageSkin, xC_B - 2, yC + 2, Color.Black);
            SafeSetPixel(imageSkin, xC_B + 2, yC - 2, Color.Black);
            SafeSetPixel(imageSkin, xC_B + 2, yC + 2, Color.Black);

            return new GraphImagePair { ImageNoSkin = imageNoSkin, ImageSkin = imageSkin };
        }

        // Функция для безопасной установки пикселей
        private void SafeSetPixel(Bitmap image, int x, int y, Color color)
        {
            if (x >= 0 && x < image.Width && y >= 0 && y < image.Height)
            {
                image.SetPixel(x, y, color);
            }
        }
    }
}
